from abc import ABC, abstractmethod

import logging
import io
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from fastcoref.coref_models.modeling_fcoref import FastCorefModel
from fastcoref.coref_models.modeling_lingmess import LingMessModel
from fastcoref.utilities.util import set_seed, create_mention_to_antecedent, create_clusters, align_to_char_level, \
    align_clusters_to_char_level
from fastcoref.utilities.collate import SegmentCollator, DynamicBatchSampler, LongformerCollator
from fastcoref.utilities import coref_dataset

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class FCorefModelArgs:
    def __init__(self, device=None):
        self.model_name_or_path = 'biu-nlp/f-coref'
        self.cache_dir = 'cache'
        self.top_lambda = 0.25
        self.max_span_length = 30
        self.ffnn_size = 1024
        self.max_segment_len = 512
        self.dropout_prob = 0.3
        self.device = device
        self.seed = 42
        self.model = None


class LingMessModelArgs:
    def __init__(self, device=None):
        self.model_name_or_path = 'biu-nlp/lingmess-coref'
        self.cache_dir = 'cache'
        self.top_lambda = 0.4
        self.max_span_length = 30
        self.ffnn_size = 2048
        self.max_segment_len = 512
        self.max_doc_len = 4096
        self.dropout_prob = 0.3
        self.device = device
        self.seed = 42
        self.model = None


class CorefResult:
    def __init__(self, text, clusters, reverse_char_map, coref_logit):
        self.text = text
        self.clusters = clusters
        self.reverse_char_map = reverse_char_map
        self.coref_logit = coref_logit

    def get_clusters(self, string=False):
        if not string:
            return self.clusters

        clusters_strings = []
        for cluster in self.clusters:
            clusters_strings.append([self.text[start:end] for start, end in cluster])

        return clusters_strings

    def get_logit(self, span_i, span_j):
        if span_i not in self.reverse_char_map:
            raise ValueError(f'span_i={span_i} is not an entity!')
        if span_j not in self.reverse_char_map:
            raise ValueError(f'span_j={span_j} is not an entity!')

        span_i_idx = self.reverse_char_map[span_i][0]
        span_j_idx = self.reverse_char_map[span_j][0]

        if span_i_idx < span_j_idx:
            return self.coref_logit[span_j_idx, span_i_idx]

        return self.coref_logit[span_i_idx, span_j_idx]


class AbstractCoref(ABC):
    def __init__(self, args_class, coref_class, device):
        self.args = args_class()

        # Setup CUDA, GPU & distributed training
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.args.device = device
        self.args.n_gpu = 1 if device.type == 'cuda' else 0
        set_seed(self.args)

        config = AutoConfig.from_pretrained(self.args.model_name_or_path, cache_dir=self.args.cache_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path, use_fast=True,
            add_prefix_space=True, cache_dir=self.args.cache_dir
        )

        self.model = coref_class.from_pretrained(
            self.args.model_name_or_path, config=config,
            cache_dir=self.args.cache_dir, args=self.args
        )
        self.model.to(device)

        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f'Model Parameters: {t_params + h_params:.1f}M, '
                    f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

    @abstractmethod
    def _prepare_batches(self, texts, max_tokens_in_batch):
        pass

    def predict(self, texts, max_tokens_in_batch=10000):
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            raise ValueError(f'texts argument expected to be a list of strings, or one single text string. provided {type(texts)}')
        dataloader = self._prepare_batches(texts, max_tokens_in_batch)

        self.model.eval()
        logger.info(f"***** Running Inference on {len(dataloader.dataset)} texts *****")

        results = []
        with tqdm(desc="Inference", total=len(dataloader.dataset)) as progress_bar:
            for idx, batch in enumerate(dataloader):
                texts = batch['text']
                tokens_to_start_char = batch['tokens_to_start_char']
                tokens_to_end_char = batch['tokens_to_end_char']
                subtoken_map = batch['subtoken_map']
                new_token_map = batch['new_token_map']

                with torch.no_grad():
                    outputs = self.model(batch, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                span_starts, span_ends, mention_logits, coref_logits = outputs_np
                doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

                for i in range(len(texts)):
                    char_map, reverse_char_map = align_to_char_level(
                        span_starts[i], span_ends[i],
                        subtoken_map[i], new_token_map[i],
                        tokens_to_start_char[i], tokens_to_end_char[i]
                    )

                    doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                    predicted_clusters = create_clusters(doc_mention_to_antecedent)
                    predicted_clusters = align_clusters_to_char_level(predicted_clusters, char_map)

                    res = CorefResult(
                        text=texts[i], clusters=predicted_clusters,
                        reverse_char_map=reverse_char_map, coref_logit=coref_logits[i]
                    )
                    results.append(res)

                progress_bar.update(n=len(batch))

        return results


class FCoref(AbstractCoref):

    def __init__(self, device='cpu'):
        super().__init__(FCorefModelArgs, FastCorefModel, device)

        self.collator = SegmentCollator(
            tokenizer=self.tokenizer, device=self.args.device,
            max_segment_len=self.args.max_segment_len
        )

    def _prepare_batches(self, texts, max_tokens_in_batch):
        f = io.StringIO(pd.DataFrame(texts, columns=['text']).to_json(orient='records', lines=True))
        dataset, dataset_files = coref_dataset.create(self.tokenizer, test_file=f,
                                                      cache_dir=self.args.cache_dir, api=True)

        dataloader = DynamicBatchSampler(
            dataset['test'],
            collator=self.collator,
            max_tokens=max_tokens_in_batch,
            max_segment_len=self.args.max_segment_len,
        )

        return dataloader


class LingMessCoref(AbstractCoref):

    def __init__(self, device='cpu'):
        super().__init__(LingMessModelArgs, LingMessModel, device)

        self.collator = LongformerCollator(
            tokenizer=self.tokenizer, device=self.args.device)

    def _prepare_batches(self, texts, max_tokens_in_batch):
        f = io.StringIO(pd.DataFrame(texts, columns=['text']).to_json(orient='records', lines=True))
        dataset, dataset_files = coref_dataset.create(self.tokenizer, test_file=f,
                                                      cache_dir=self.args.cache_dir, api=True)

        dataloader = DynamicBatchSampler(
            dataset['test'],
            collator=self.collator,
            max_tokens=max_tokens_in_batch,
            max_segment_len=self.args.max_segment_len,
            max_doc_len=self.args.max_doc_len
        )

        return dataloader
