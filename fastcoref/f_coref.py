from abc import ABC, abstractmethod

import logging
import io
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from datasets import Dataset
import spacy
from spacy.cli import download

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


class CorefArgs:
    def __init__(self, model_name_or_path, top_lambda, ffnn_size, max_segment_len, max_doc_len):
        self.model_name_or_path = model_name_or_path
        self.cache_dir = 'cache'
        self.top_lambda = top_lambda
        self.max_span_length = 30
        self.ffnn_size = ffnn_size
        self.max_segment_len = max_segment_len
        self.max_doc_len = max_doc_len
        self.dropout_prob = 0.3
        self.device = None
        self.seed = 42


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


COREF_CLASSES = {
    'FCoref': FastCorefModel,
    'LingMessCoref': LingMessModel
}

COREF_COLLATORS = {
    'FCoref': SegmentCollator,
    'LingMessCoref': LongformerCollator
}

COREF_ARGS = {
    'FCoref': CorefArgs(model_name_or_path='biu-nlp/f-coref', top_lambda=0.25,
                        ffnn_size=1024, max_segment_len=512, max_doc_len=None),

    'LingMessCoref': CorefArgs(model_name_or_path='biu-nlp/lingmess-coref', top_lambda=0.4,
                               ffnn_size=2048, max_segment_len=512, max_doc_len=4096),
}


class AutoCoref(ABC):
    def __init__(self, model_type, device):
        self.args = COREF_ARGS[model_type]
        COREF_CLASS = COREF_CLASSES[model_type]

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

        self.collator = COREF_COLLATORS[model_type](
            tokenizer=self.tokenizer, device=self.args.device,
            max_segment_len=self.args.max_segment_len
        )

        self.model = COREF_CLASS.from_pretrained(
            self.args.model_name_or_path, config=config,
            cache_dir=self.args.cache_dir, args=self.args
        )
        self.model.to(device)

        t_params, h_params = [p / 1000000 for p in self.model.num_parameters()]
        logger.info(f'Model Parameters: {t_params + h_params:.1f}M, '
                    f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

        try:
            self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        except OSError:
            # TODO: this is a workaround it is not clear how to add "en_core_web_sm" to setup.py
            download('en_core_web_sm')
            self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

    def _prepare_dataset(self, texts):
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
            for idx, batch in enumerate(dataloader):
                texts = batch['text']
                subtoken_map = batch['subtoken_map']
                new_token_map = batch['new_token_map']
                token_to_char = batch['token_to_char']

                with torch.no_grad():
                    outputs = self.model(batch, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                span_starts, span_ends, mention_logits, coref_logits = outputs_np
                doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

                for i in range(len(texts)):
                    doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                    predicted_clusters = create_clusters(doc_mention_to_antecedent)

                    char_map, reverse_char_map = align_to_char_level(
                        span_starts[i], span_ends[i], subtoken_map[i], new_token_map[i], token_to_char[i]
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

        dataset = self._prepare_dataset(texts=texts)
        dataloader = self._prepare_batches(dataset, max_tokens_in_batch)

        if is_str:
            return self._inference(dataloader=dataloader)[0]
        return self._inference(dataloader=dataloader)


class FCoref(AutoCoref):

    def __init__(self, device='cpu'):
        super().__init__('FCoref', device)


def encode(batch, tokenizer, nlp):
    tokenized_texts = tokenize_with_spacy(batch['text'], nlp)
    encoded_batch = tokenizer(tokenized_texts['tokens'], add_special_tokens=True, is_split_into_words=True)
    return {
        'input_ids': encoded_batch['input_ids'],
        'attention_mask': encoded_batch['attention_mask'],
        'length': [len(ids) for ids in encoded_batch['input_ids']],
        'new_token_map': [list(range(len(tokens))) for tokens in tokenized_texts['tokens']],       # preparation for speaker in the future
        'subtoken_map': [enc.word_ids for enc in encoded_batch.encodings],
        **tokenized_texts
    }

def tokenize_with_spacy(texts, nlp):
    def handle_doc(doc):
        tokens = []
        token_to_char = []
        for tok in doc:
            tokens.append(tok.text)
            token_to_char.append((tok.idx, tok.idx + len(tok.text)))
        return tokens, token_to_char

    tokenized_texts = {'tokens': [], 'token_to_char': []}
    for doc in tqdm(nlp.pipe(texts)):
        tokens, token_to_char = handle_doc(doc)
        tokenized_texts['tokens'].append(tokens)
        tokenized_texts['token_to_char'].append(token_to_char)

    return tokenized_texts


class LingMessCoref(AutoCoref):

    def __init__(self, device='cpu'):
        super().__init__('LingMessCoref', device)




