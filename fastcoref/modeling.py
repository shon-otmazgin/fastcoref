import json
from abc import ABC

import logging
from typing import List, Union

import torch
import transformers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from datasets import Dataset
import spacy
from spacy.cli import download
from spacy.language import Language

from fastcoref.coref_models.modeling_fcoref import FCorefModel
from fastcoref.coref_models.modeling_lingmess import LingMessModel
from fastcoref.utilities.util import set_seed, create_mention_to_antecedent, create_clusters, align_to_char_level, encode
from fastcoref.utilities.collate import LeftOversCollator, DynamicBatchSampler, PadCollator

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class CorefResult:
    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit, text_idx):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit
        self.text_idx = text_idx

    def get_clusters(self, as_strings=True):
        if not as_strings:
            return [[self.char_map[mention][1] for mention in cluster] for cluster in self.clusters]

        return [[self.text[self.char_map[mention][1][0]:self.char_map[mention][1][1]]
                 for mention in cluster if None not in self.char_map[mention]] for cluster in self.clusters]

    def get_logit(self, span_i, span_j):
        if span_i not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_i[0]:span_i[1]]}" is not an entity in this model!')
        if span_j not in self.reverse_char_map:
            raise ValueError(f'span_i="{self.text[span_j[0]:span_j[1]]}" is not an entity in this model!')

        span_i_idx = self.reverse_char_map[span_i][0]   # 0 is to get the span index
        span_j_idx = self.reverse_char_map[span_j][0]

        if span_i_idx < span_j_idx:
            return self.coref_logit[span_j_idx, span_i_idx]

        return self.coref_logit[span_i_idx, span_j_idx]

    def __str__(self):
        if len(self.text) > 50:
            text_to_print = f'{self.text[:50]}...'
        else:
            text_to_print = self.text
        return f'CorefResult(text="{text_to_print}", clusters={self.get_clusters()})'

    def __repr__(self):
        return self.__str__()


class CorefModel(ABC):
    def __init__(self, model_name_or_path, coref_class, collator_class, enable_progress_bar, device=None, nlp="en_core_web_sm"):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.seed = 42
        self._set_device()
        self.enable_progress_bar = enable_progress_bar

        config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.max_segment_len = config.coref_head['max_segment_len']
        self.max_doc_len = config.coref_head['max_doc_len'] if 'max_doc_len' in config.coref_head else None

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=True,
            add_prefix_space=True, verbose=False
        )

        if collator_class == PadCollator:
            self.collator = PadCollator(tokenizer=self.tokenizer, device=self.device)
        elif collator_class == LeftOversCollator:
            self.collator = LeftOversCollator(
                tokenizer=self.tokenizer, device=self.device,
                max_segment_len=config.coref_head['max_segment_len']
            )
        else:
            raise NotImplementedError(f"Class collator {type(collator_class)} is not supported! "
                                      f"only LeftOversCollator or PadCollator supported")
        if nlp is None:
            self.nlp = None
            logger.warning(
                "You didn't specify a spacy model, you'll need to provide tokenized text in the `predict` function."
            )
        elif isinstance(nlp, Language):
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load(nlp, exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
            except OSError:
                # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
                download(nlp)
                self.nlp = spacy.load(nlp, exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        self.model, loading_info = coref_class.from_pretrained(
            self.model_name_or_path, config=config,
            output_loading_info=True
        )
        self.model.to(self.device)

        for key, val in loading_info.items():
            logger.info(f'{key}: {list(set(val) - set(["longformer.embeddings.position_ids"]))}')
        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f'Model Parameters: {t_params + h_params:.1f}M, '
                    f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

        set_seed(self)
        transformers.logging.set_verbosity_error()

    def _set_device(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.n_gpu = torch.cuda.device_count()

    def _create_dataset(self, texts, is_split_into_words):
        logger.info(f'Tokenize {len(texts)} inputs...')

        # Save original text ordering for later use
        dataset = {'text': texts, 'idx': range(len(texts))}
        if is_split_into_words:
            dataset['tokens'] = texts

        dataset = Dataset.from_dict(dataset)
        dataset = dataset.map(
            encode, batched=True, batch_size=10000,
            fn_kwargs={'tokenizer': self.tokenizer, 'nlp': self.nlp if not is_split_into_words else None}
        )

        return dataset

    def _prepare_batches(self, dataset, max_tokens_in_batch):
        dataloader = DynamicBatchSampler(
            dataset,
            collator=self.collator,
            max_tokens=max_tokens_in_batch,
            max_segment_len=self.max_segment_len,
            max_doc_len=self.max_doc_len
        )

        return dataloader

    def _batch_inference(self, batch):
        texts = batch['text']
        subtoken_map = batch['subtoken_map']
        token_to_char = batch['offset_mapping']
        idxs = batch['idx']
        with torch.no_grad():
            outputs = self.model(batch, return_all_outputs=True)

        outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

        span_starts, span_ends, mention_logits, coref_logits = outputs_np
        doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

        results = []

        for i in range(len(texts)):
            doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
            predicted_clusters = create_clusters(doc_mention_to_antecedent)

            char_map, reverse_char_map = align_to_char_level(
                span_starts[i], span_ends[i], token_to_char[i], subtoken_map[i]
            )

            result = CorefResult(
                text=texts[i], clusters=predicted_clusters,
                char_map=char_map, reverse_char_map=reverse_char_map,
                coref_logit=coref_logits[i], text_idx=idxs[i]
            )

            results.append(result)

        return results

    def _inference(self, dataloader):
        self.model.eval()
        logger.info(f"***** Running Inference on {len(dataloader.dataset)} texts *****")

        results = []
        if self.enable_progress_bar:
            with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
                for batch in dataloader:
                    results.extend(self._batch_inference(batch))
                    progress_bar.update(n=len(batch['text']))
        else:
            for batch in dataloader:
                results.extend(self._batch_inference(batch))

        return sorted(results, key=lambda res: res.text_idx)

    def predict(self,
                texts: Union[str, List[str], List[List[str]]],  # similar to huggingface tokenizer inputs
                is_split_into_words: bool = False,
                max_tokens_in_batch: int = 10000,
                output_file: str = None):
        """
        texts (str, List[str], List[List[str]]) — The sequence or batch of sequences to be encoded.
        Each sequence can be a string or a list of strings (pretokenized string).
        If the sequences are provided as list of strings (pretokenized), you must set is_split_into_words=True
        (to lift the ambiguity with a batch of sequences).
        is_split_into_words - indicate if the texts input is tokenized
        """

        # Input type checking for clearer error
        def _is_valid_text_input(texts, is_split_into_words):
            if isinstance(texts, str) and not is_split_into_words:
                # Strings are fine
                return True
            elif isinstance(texts, (list, tuple)):
                # List are fine as long as they are...
                if len(texts) == 0:
                    # ... empty
                    return True
                elif all([isinstance(t, str) for t in texts]):
                    # ... list of strings
                    return True
                elif all([isinstance(t, (list, tuple)) for t in texts]):
                    # ... list with an empty list or with a list of strings
                    return len(texts[0]) == 0 or isinstance(texts[0][0], str)
                else:
                    return False
            else:
                return False

        if not _is_valid_text_input(texts, is_split_into_words):
            raise ValueError(
                "text input must be of type `str` (single example), `List[str]` (batch or single pretokenized example) "
                "or `List[List[str]]` (batch of pretokenized examples)."
            )

        if not is_split_into_words and not self.nlp:
            raise ValueError(
                "Model initialized with no nlp component for tokenizing the text, please pass pretokenized text,"
                "or initialize the model with an nlp component."
            )

        if is_split_into_words:
            is_batched = isinstance(texts, (list, tuple)) and texts and isinstance(texts[0], (list, tuple))
        else:
            is_batched = isinstance(texts, (list, tuple))

        if not is_batched:
            texts = [texts]

        dataset = self._create_dataset(texts, is_split_into_words)
        dataloader = self._prepare_batches(dataset, max_tokens_in_batch)

        preds = self._inference(dataloader)
        if output_file is not None:
            with open(output_file, 'w') as f:
                data = [{'text': p.text,
                         'clusters': p.get_clusters(as_strings=False),
                         'clusters_strings': p.get_clusters(as_strings=True)}
                        for p in preds]
                f.write('\n'.join(map(json.dumps, data)))
        if not is_batched:
            return preds[0]
        return preds


class FCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/f-coref', device=None, nlp="en_core_web_sm", enable_progress_bar=True):
        super().__init__(model_name_or_path, FCorefModel, LeftOversCollator, enable_progress_bar, device, nlp)


class LingMessCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/lingmess-coref', device=None, nlp="en_core_web_sm", enable_progress_bar=True):
        super().__init__(model_name_or_path, LingMessModel, PadCollator, enable_progress_bar, device, nlp)
