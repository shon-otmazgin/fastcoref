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


class CorefArgs:
    def __init__(self,
                 model_name_or_path: str,
                 output_dir: str = None,
                 overwrite_output_dir: bool = False,
                 train_file: str = None,
                 dev_file: str = None,
                 test_file: str = None,
                 output_file: str = None,
                 learning_rate: float = 1e-5,
                 head_learning_rate: float = 3e-4,
                 dropout_prob: float = 0.3,
                 weight_decay: float = 0.01,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.98,
                 adam_epsilon: float = 1e-6,
                 epochs: float = 3,
                 ffnn_size: int = 1024,
                 logging_steps: int = 500,
                 eval_steps: int = 500,
                 device: str = None,
                 seed: int = 42,
                 max_span_length: int = 30,
                 top_lambda: float = 0.4,
                 cache_dir: str = 'cache',
                 experiment_name: str = None,
                 max_segment_len: int = 512,
                 max_doc_len: int = None,
                 max_tokens_in_batch: int = 5000):
        self.max_tokens_in_batch = max_tokens_in_batch
        self.max_doc_len = max_doc_len
        self.max_segment_len = max_segment_len
        self.experiment_name = experiment_name
        self.cache_dir = cache_dir
        self.top_lambda = top_lambda
        self.device = device
        self.n_gpu = None
        self.max_span_length = max_span_length
        self.seed = seed
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.ffnn_size = ffnn_size
        self.epochs = epochs
        self.adam_epsilon = adam_epsilon
        self.adam_beta2 = adam_beta2
        self.adam_beta1 = adam_beta1
        self.weight_decay = weight_decay
        self.dropout_prob = dropout_prob
        self.head_learning_rate = head_learning_rate
        self.learning_rate = learning_rate
        self.output_file = output_file
        self.test_file = test_file
        self.dev_file = dev_file
        self.train_file = train_file
        self.overwrite_output_dir = overwrite_output_dir
        self.output_dir = output_dir
        self.model_name_or_path = model_name_or_path


class CorefResult:
    def __init__(self, text, clusters, char_map, reverse_char_map, coref_logit):
        self.text = text
        self.clusters = clusters
        self.char_map = char_map
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit

    def get_clusters(self, as_strings=True):
        if not as_strings:
            return [[self.char_map[mention][1] for mention in cluster] for cluster in self.clusters]

        return [[self.text[self.char_map[mention][1][0]:self.char_map[mention][1][1]] for mention in cluster]
                for cluster in self.clusters]

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
    def __init__(self, coref_class, collator_class, args):
        self.args = args
        self._set_device()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, use_fast=True,
            add_prefix_space=True, cache_dir=self.args.cache_dir, verbose=False
        )

        if collator_class == PadCollator:
            self.collator = PadCollator(tokenizer=self.tokenizer, device=self.args.device)
        elif collator_class == LeftOversCollator:
            self.collator = LeftOversCollator(
                tokenizer=self.tokenizer, device=self.args.device,
                max_segment_len=self.args.max_segment_len
            )
        else:
            raise NotImplementedError(f"Class collator {type(collator_class)} is not supported! "
                                      f"only LeftOversCollator or PadCollator supported")

        try:
            self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        except OSError:
            # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
            download('en_core_web_sm')
            self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        config = AutoConfig.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)
        self.model, loading_info = coref_class.from_pretrained(
            self.args.model_name_or_path, config=config,
            cache_dir=self.args.cache_dir, args=self.args,
            output_loading_info=True
        )
        self.model.to(self.args.device)

        for key, val in loading_info.items():
            logger.info(f'{key}: {list(set(val) - set(["longformer.embeddings.position_ids"]))}')
        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f'Model Parameters: {t_params + h_params:.1f}M, '
                    f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

        set_seed(self.args)
        transformers.logging.set_verbosity_error()

    def _set_device(self):
        if self.args.device is None:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.device = torch.device(self.args.device)
        self.args.n_gpu = torch.cuda.device_count()

    def _create_dataset(self, texts):
        logger.info(f'Tokenize {len(texts)} texts...')

        dataset = Dataset.from_dict({'text': texts})
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
            max_segment_len=self.args.max_segment_len,
            max_doc_len=self.args.max_doc_len
        )

        return dataloader

    def _inference(self, dataloader):
        self.model.eval()
        logger.info(f"***** Running Inference on {len(dataloader.dataset)} texts *****")

        results = []
        with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
            for batch in dataloader:
                texts = batch['text']
                subtoken_map = batch['subtoken_map']
                token_to_char = batch['offset_mapping']

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
                        text=texts[i], clusters=predicted_clusters,
                        char_map=char_map, reverse_char_map=reverse_char_map,
                        coref_logit=coref_logits[i]
                    )
                    results.append(res)

                progress_bar.update(n=len(texts))

        return results

    def predict(self, texts, max_tokens_in_batch=10000):
        is_str = False
        if isinstance(texts, str):
            texts = [texts]
            is_str = True
        if not isinstance(texts, list):
            raise ValueError(f'texts argument expected to be a list of strings, or one single text string. provided {type(texts)}')

        dataset = self._create_dataset(texts=texts)
        dataloader = self._prepare_batches(dataset, max_tokens_in_batch)

        preds = self._inference(dataloader=dataloader)
        if is_str:
            return preds[0]
        return preds


class FCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/f-coref', top_lambda=0.25, max_span_length=30, ffnn_size=1024,
                 max_segment_len=512, max_doc_len=None, cache_dir='cache', device=None):
        args = CorefArgs(
            model_name_or_path=model_name_or_path, top_lambda=top_lambda, max_span_length=max_span_length,
            ffnn_size=ffnn_size, max_segment_len=max_segment_len, max_doc_len=max_doc_len,
            cache_dir=cache_dir, device=device
        )
        super().__init__(FCorefModel, LeftOversCollator, args)


class LingMessCoref(CorefModel):
    def __init__(self, model_name_or_path='biu-nlp/lingmess-coref', top_lambda=0.4, max_span_length=30, ffnn_size=2048,
                 max_segment_len=512, max_doc_len=4096, cache_dir='cache', device=None):
        args = CorefArgs(
            model_name_or_path=model_name_or_path, top_lambda=top_lambda, max_span_length=max_span_length,
            ffnn_size=ffnn_size, max_segment_len=max_segment_len, max_doc_len=max_doc_len,
            cache_dir=cache_dir, device=device
        )
        super().__init__(LingMessModel, PadCollator, args)




