import logging
import numpy as np
import torch
import spacy
from dataclasses import dataclass
from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, AutoTokenizer

from coref_models.modeling_fcoref import FCorefModel

from utilities import coref_dataset
from utilities.collate import DynamicBatchSampler, LeftOversCollator
from utilities.consts import SUPPORTED_MODELS
from utilities.metrics import MentionEvaluator, CorefEvaluator
from utilities.util import set_seed, create_mention_to_antecedent, create_clusters, \
    output_evaluation_metrics, update_metrics

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


@dataclass
class TrainingArgs:
    model_name_or_path: str
    output_dir: str = None
    overwrite_output_dir: bool = False
    output_file: str = None
    learning_rate: float = 1e-5
    head_learning_rate: float = 3e-4
    dropout_prob: float = 0.3
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-6
    epochs: float = 3
    ffnn_size: int = 1024
    logging_steps: int = 500
    eval_steps: int = 500
    seed: int = 42
    max_span_length: int = 30
    top_lambda: float = 0.4
    cache_dir: str = 'cache'
    experiment_name: str = None
    max_segment_len: int = 512
    max_doc_len: int = None
    max_tokens_in_batch: int = 5000
    device: str = None
    n_gpu: int = 0


def _load_f_coref_model(args):
    logger.info(f'Loading FCoref model with underlying transformer {args.model_name_or_path}')

    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    config.coref_head = {
        "max_span_length": args.max_span_length,
        "top_lambda": args.top_lambda,
        "ffnn_size": args.ffnn_size,
        "dropout_prob": args.dropout_prob,
        "max_segment_len": args.max_segment_len
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=True, add_prefix_space=True, cache_dir=args.cache_dir
    )

    model = FCorefModel.from_pretrained(
        args.model_name_or_path, output_loading_info=False,
        config=config, cache_dir=args.cache_dir
    )

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'FCoref Parameters: {t_params + h_params:.1f}M, '
                f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

    if model.base_model_prefix not in SUPPORTED_MODELS:
        raise NotImplementedError(f'Not supporting {model.base_model_prefix}, choose one of {SUPPORTED_MODELS}')

    return model, tokenizer


class CorefTrainer:
    def __init__(self, args: TrainingArgs, train_file, dev_file=None):
        transformers.logging.set_verbosity_error()
        self.args = args
        self._set_device()

        self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        self.model, self.tokenizer = _load_f_coref_model(self.args)
        self.model.to(self.device)

        self.collator = LeftOversCollator(
            tokenizer=self.tokenizer,
            device=self.device,
            max_segment_len=self.args.max_segment_len
        )

        self.train_dataset = coref_dataset.create(
            file=train_file, tokenizer=self.tokenizer, nlp=self.nlp
        )
        self.train_sampler = DynamicBatchSampler(
            dataset=self.train_dataset,
            collator=self.collator,
            max_tokens=self.args.max_tokens_in_batch,
            max_segment_len=self.args.max_segment_len
        )

        self.dev_dataset, self.train_sampler = None, None
        if dev_file is not None:
            self.dev_dataset = coref_dataset.create(file=dev_file, tokenizer=self.tokenizer, nlp=self.nlp)
            self.dev_sampler = DynamicBatchSampler(
                dataset=self.dev_dataset,
                collator=self.collator,
                max_tokens=self.args.max_tokens_in_batch * 2,
                max_segment_len=self.args.max_segment_len
            )

        set_seed(self.args)

    def _set_device(self):
        # Setup CUDA, GPU & distributed training
        if self.args.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(self.args.device)
        # TODO: this is not true
        self.args.n_gpu = torch.cuda.device_count()

    def train(self):
        pass

    def evaluate(self, prefix=''):
        self.model.eval()

        logger.info(f"***** Running Inference on {len(self.dev_sampler.dataset)} documents *****")

        metrics_dict = {'loss': 0., 'post_pruning': MentionEvaluator(), 'mentions': MentionEvaluator(),
                        'coref': CorefEvaluator()}
        doc_to_tokens = {}
        doc_to_subtoken_map = {}
        doc_to_new_word_map = {}
        doc_to_prediction = {}

        with tqdm(desc="Inference", total=len(self.dev_sampler.dataset)) as progress_bar:
            for idx, batch in enumerate(self.dev_sampler):
                doc_keys = batch['doc_key']
                tokens = batch['tokens']
                subtoken_map = batch['subtoken_map']
                new_token_map = batch['new_token_map']
                gold_clusters = batch['gold_clusters']

                with torch.no_grad():
                    outputs = self.model(batch, gold_clusters=gold_clusters, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                gold_clusters = gold_clusters.cpu().numpy()
                loss, span_starts, span_ends, mention_logits, coref_logits = outputs_np
                metrics_dict['loss'] += loss.item()

                doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

                for i, doc_key in enumerate(doc_keys):
                    doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                    predicted_clusters = create_clusters(doc_mention_to_antecedent)

                    doc_to_prediction[doc_key] = predicted_clusters
                    doc_to_tokens[doc_key] = tokens[i]
                    doc_to_subtoken_map[doc_key] = subtoken_map[i]
                    doc_to_new_word_map[doc_key] = new_token_map[i]

                    update_metrics(metrics_dict, span_starts[i], span_ends[i], gold_clusters[i], predicted_clusters)

                progress_bar.update(n=len(doc_keys))

        results = output_evaluation_metrics(
            metrics_dict=metrics_dict, prefix=prefix
        )

        return results

    def push_to_hub(self, repo_name, organization=None):
        self.model.push_to_hub(repo_name, organization=organization, use_temp_dir=True)
        self.tokenizer.push_to_hub(repo_name, organization=organization, use_temp_dir=True)


if __name__ == '__main__':
    args = TrainingArgs(
        model_name_or_path='biu-nlp/f-coref',
        device='cuda:2'
    )
    trainer = CorefTrainer(
        args=args,
        # train_file='/Users/sotmazgin/Desktop/fastcoref/dev.english.jsonlines',
        # dev_file='/Users/sotmazgin/Desktop/fastcoref/dev.english.jsonlines',
        train_file='/home/nlp/shon711/lingmess-coref/prepare_ontonotes/dev.english.jsonlines',
        dev_file='/home/nlp/shon711/lingmess-coref/prepare_ontonotes/dev.english.jsonlines'
    )
    trainer.evaluate()