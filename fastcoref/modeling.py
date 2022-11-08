import json
from abc import ABC

import logging
import torch
import transformers
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from datasets import Dataset
import spacy
from spacy.cli import download

from fastcoref.coref_models.modeling_fcoref import FCorefModel
from fastcoref.coref_models.modeling_lingmess import LingMessModel
from fastcoref.utilities.util import set_seed, create_mention_to_antecedent, create_clusters, align_to_char_level, encode
from fastcoref.utilities.collate import LeftOversCollator, DynamicBatchSampler, PadCollator

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class CorefResult:
    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit, text_idx, tokenized_text=None, subtoken_map=None):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit
        self.text_idx = text_idx
        self.tokenized_text = tokenized_text
        self.subtoken_map = subtoken_map  # subtoken index -> token index
        # (token start offset, token end offset) -> subtoken span index
        self.reverse_subtoken_map = None
        if subtoken_map:
            self.reverse_subtoken_map = dict()
            for (stok_start, stok_end), (span_i, _) in char_map.items():
                self.reverse_subtoken_map[(subtoken_map[stok_start], subtoken_map[stok_end])] = span_i

    def get_clusters(self, as_strings=True):
        if not as_strings:
            return [[self.char_map[mention][1] for mention in cluster] for cluster in self.clusters]

        return [[self.text[self.char_map[mention][1][0]:self.char_map[mention][1][1]] for mention in cluster]
                for cluster in self.clusters]

    def get_clusters_tokenized(self):
        if not self.tokenized_text:
            raise ValueError("Tokenized version is only if you called predict with pretokenized text")
        return [[(self.subtoken_map[mention[0]], self.subtoken_map[mention[1]]) for mention in cluster] for cluster in self.clusters]

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

    def get_logit_tokenized(self, span_token_i, span_token_j):
        if not self.tokenized_text or not self.reverse_subtoken_map:
            raise ValueError("Tokenized version is only if you called predict with pretokenized text")
        if span_token_i not in self.reverse_subtoken_map:
            raise ValueError(f'span_token_i="{self.text[span_token_i[0]:span_token_i[1]]}" is not an entity in this model!')
        if span_token_j not in self.reverse_subtoken_map:
            raise ValueError(f'span_token_j="{self.text[span_token_j[0]:span_token_j[1]]}" is not an entity in this model!')
        return self.get_logit[
            self.reverse_subtoken_map[span_token_i],
            self.reverse_subtoken_map[span_token_j],
        ]

    def __str__(self):
        if len(self.text) > 50:
            text_to_print = f'{self.text[:50]}...'
        else:
            text_to_print = self.text
        return f'CorefResult(text="{text_to_print}", clusters={self.get_clusters()})'

    def __repr__(self):
        return self.__str__()


class CorefModel(ABC):
    def __init__(self, model_name_or_path, coref_class, collator_class, device=None, nlp=None):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.seed = 42
        self._set_device()

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
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
            except OSError:
                # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
                download('en_core_web_sm')
                self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

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

    def _create_dataset(self, texts, tokenized_texts=None):
        logger.info(f'Tokenize {len(texts)} texts...')
        # Save original text ordering for later use
        dataset = Dataset.from_dict({'text': texts, 'tokenized_text': tokenized_texts, 'idx':range(len(texts))})
        dataset = dataset.map(
            encode, batched=True, batch_size=10000,
            fn_kwargs={'tokenizer': self.tokenizer, 'nlp': self.nlp}
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

    def _inference(self, dataloader):
        self.model.eval()
        logger.info(f"***** Running Inference on {len(dataloader.dataset)} texts *****")

        results = []
        with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
            for batch in dataloader:
                texts = batch['text']
                tokenized_texts = batch['tokenized_text']
                subtoken_map = batch['subtoken_map']
                token_to_char = batch['offset_mapping']
                idxs = batch['idx']
                with torch.no_grad():
                    outputs = self.model(batch, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                span_starts, span_ends, mention_logits, coref_logits = outputs_np
                doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

                for i in range(len(texts)):
                    doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                    predicted_clusters = create_clusters(doc_mention_to_antecedent)

                    char_map, reverse_char_map = align_to_char_level(
                        span_starts[i], span_ends[i], token_to_char[i], subtoken_map[i]
                    )

                    res = CorefResult(
                        text=texts[i],
                        clusters=predicted_clusters,
                        char_map=char_map,
                        reverse_char_map=reverse_char_map,
                        coref_logit=coref_logits[i],
                        text_idx=idxs[i],
                        tokenized_text=tokenized_texts[i] if tokenized_texts else None,
                        subtoken_map=subtoken_map[i],
                    )
                    results.append(res)

                progress_bar.update(n=len(texts))

        return sorted(results, key=lambda res: res.text_idx)

    def predict(self, texts, tokenized_texts=None, max_tokens_in_batch=10000, output_file=None):
        is_str = False
        if isinstance(texts, str):
            texts = [texts]
            is_str = True
        elif texts is None:
            if tokenized_texts is None:
                raise ValueError('either texts or tokenized_texts arguments expected to have a value')
            texts = [" ".join(tokenized_text) for tokenized_text in tokenized_texts]
        elif not isinstance(texts, list):
            raise ValueError(f'texts argument expected to be a list of strings, or one single text string. provided {type(texts)}')

        dataset = self._create_dataset(texts=texts, tokenized_texts=tokenized_texts)
        dataloader = self._prepare_batches(dataset, max_tokens_in_batch)

        preds = self._inference(dataloader=dataloader)
        if output_file is not None:
            with open(output_file, 'w') as f:
                data = [{'text': p.text,
                         'clusters': p.get_clusters(as_strings=False),
                         'clusters_strings': p.get_clusters(as_strings=True)}
                        for p in preds]
                f.write('\n'.join(map(json.dumps, data)))
        if is_str:
            return preds[0]
        return preds


class FCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/f-coref', device=None, nlp=None):
        super().__init__(model_name_or_path, FCorefModel, LeftOversCollator, device, nlp)


class LingMessCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/lingmess-coref', device=None, nlp=None):
        super().__init__(model_name_or_path, LingMessModel, PadCollator, device, nlp)




