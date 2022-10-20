import logging
import torch
import spacy
from dataclasses import dataclass

import transformers
from transformers import AutoConfig, AutoTokenizer

from models.modeling_fcoref import FCorefModel
from models.modeling_lingmess import LingMessModel

# Setup logging
from utilities import coref_dataset
from utilities.consts import SUPPORTED_MODELS
from utilities.util import set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


@dataclass
class TrainingArgs:
    model_name_or_path: str
    output_dir: str = None
    overwrite_output_dir: bool = False
    train_file: str = None
    dev_file: str = None
    test_file: str = None
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


def _load_annotator_model():
    annotator_coref_model = 'biu-nlp/lingmess-coref'

    logger.info(f'Loading LingMess model for annotation')

    config = AutoConfig.from_pretrained(annotator_coref_model, cache_dir=args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        annotator_coref_model, use_fast=True, add_prefix_space=True, cache_dir=args.cache_dir
    )

    model = LingMessModel.from_pretrained(
        annotator_coref_model, output_loading_info=False,
        config=config, cache_dir=args.cache_dir
    )

    t_params, h_params = [p / 1000000 for p in model.num_parameters()]
    logger.info(f'LingMess Parameters: {t_params + h_params:.1f}M, '
                f'Transformer: {t_params:.1f}M, Coref head: {h_params:.1f}M')

    return model, tokenizer


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
    def __init__(self, args: TrainingArgs, annotate=False):
        transformers.logging.set_verbosity_error()

        if self.args.train_file is None:
            raise ValueError(f'Train file cannot be None!')

        self.args = args
        self.nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        self.model, self.tokenizer = _load_f_coref_model(self.args)
        self._set_device()

        self.dataset_to_annotate = None
        if annotate:
            self.annotator_model, self.annotator_tokenizer = _load_annotator_model()
            self.dataset_to_annotate = coref_dataset.create(
                file=args.train_file, tokenizer=self.annotator_tokenizer, nlp=self.nlp
            )
            # run evaluate.

        if self.dataset_to_annotate is None:
            self.train_dataset = coref_dataset.create(
                file=args.train_file, tokenizer=self.tokenizer, nlp=self.nlp
            )
        else:
            pass
            # create train dataset from dataset_to_annotate

        self.dev_dataset = None
        if self.args.dev_file is not None:
            self.dev_dataset = coref_dataset.create(file=args.dev_file, tokenizer=self.tokenizer, nlp=self.nlp)

        set_seed(self.args)

    def _set_device(self):
        # Setup CUDA, GPU & distributed training
        if self.args.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device)
        self.n_gpu = torch.cuda.device_count()

    def train(self):
        pass

    def evaluate(self):
        pass

    def push_to_hub(self, repo_name, organization=None):
        self.model.push_to_hub(repo_name, organization=organization, use_temp_dir=True)
        self.tokenizer.push_to_hub(repo_name, organization=organization, use_temp_dir=True)


if __name__ == '__main__':
    args = TrainingArgs(
        model_name_or_path='distilroberta-base',
        train_file='/Users/sotmazgin/Desktop/fastcoref/toy_dataset.jsonlines'
    )
    CorefTrainer(args)