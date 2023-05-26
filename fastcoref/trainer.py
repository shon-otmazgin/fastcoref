import os
import logging
import numpy as np
import torch
import spacy
from spacy.language import Language
from spacy.cli import download
from dataclasses import dataclass

from torch.optim.adamw import AdamW
from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup

from fastcoref.coref_models.modeling_fcoref import FCorefModel

from fastcoref.utilities import coref_dataset
from fastcoref.utilities.collate import DynamicBatchSampler, LeftOversCollator
from fastcoref.utilities.consts import SUPPORTED_MODELS
from fastcoref.utilities.metrics import MentionEvaluator, CorefEvaluator
from fastcoref.utilities.util import set_seed, create_mention_to_antecedent, create_clusters, \
    output_evaluation_metrics, update_metrics, save_all

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - \t %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


@dataclass
class TrainingArgs:
    model_name_or_path: str
    output_dir: str
    overwrite_output_dir: bool = False
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
    max_segment_len: int = 512
    max_doc_len: int = None
    max_tokens_in_batch: int = 5000
    device: str = None


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
    def __init__(self, args: TrainingArgs, train_file, dev_file=None, test_file=None, nlp=None):
        import wandb

        transformers.logging.set_verbosity_error()
        self.args = args
        wandb.init(project=self.args.output_dir, config=self.args)
        self.wandb_logger = wandb.log
        self.wandb_runner = wandb.run

        self._set_device()
        self.nlp = nlp if isinstance(nlp, Language) else spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])

        self.model, self.tokenizer = _load_f_coref_model(self.args)
        self.model.to(self.device)

        self.collator = LeftOversCollator(
            tokenizer=self.tokenizer,
            device=self.device,
            max_segment_len=self.args.max_segment_len
        )

        self.train_dataset, self.train_sampler = self._get_sampler(train_file)

        self.dev_dataset, self.dev_sampler = None, None
        if dev_file is not None:
            self.dev_dataset, self.dev_sampler = self._get_sampler(dev_file)

        self.test_dataset, self.test_sampler = None, None
        if test_file is not None:
            self.test_dataset, self.test_sampler = self._get_sampler(test_file)

        set_seed(self.args)

    def _get_sampler(self, file):
        dataset = coref_dataset.create(
            file=file, tokenizer=self.tokenizer, nlp=self.nlp
        )
        sampler = DynamicBatchSampler(
            dataset=dataset,
            collator=self.collator,
            max_tokens=self.args.max_tokens_in_batch,
            max_segment_len=self.args.max_segment_len
        )

        return dataset, sampler

    def _set_device(self):
        # Setup CUDA, GPU & distributed training
        if self.args.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = torch.device(self.args.device)
        # TODO: this is not true
        self.args.n_gpu = torch.cuda.device_count()

    def train(self):
        """ Train the model """

        # we create batches beacuse the sampler is generating batches of sorted docs -> to avoid many pad tokens
        # so, we need to shuffle the batches somehow.
        train_batches = coref_dataset.create_batches(self.train_sampler, shuffle=True)

        t_total = len(train_batches) * self.args.epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        head_params = ['coref', 'mention', 'antecedent']

        model_decay = [p for n, p in self.model.named_parameters() if
                       not any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        model_no_decay = [p for n, p in self.model.named_parameters() if
                          not any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]
        head_decay = [p for n, p in self.model.named_parameters() if
                      any(hp in n for hp in head_params) and not any(nd in n for nd in no_decay)]
        head_no_decay = [p for n, p in self.model.named_parameters() if
                         any(hp in n for hp in head_params) and any(nd in n for nd in no_decay)]

        head_learning_rate = self.args.head_learning_rate if self.args.head_learning_rate else self.args.learning_rate
        optimizer_grouped_parameters = [
            {'params': model_decay, 'lr': self.args.learning_rate, 'weight_decay': self.args.weight_decay},
            {'params': model_no_decay, 'lr': self.args.learning_rate, 'weight_decay': 0.0},
            {'params': head_decay, 'lr': head_learning_rate, 'weight_decay': self.args.weight_decay},
            {'params': head_no_decay, 'lr': head_learning_rate, 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          betas=(self.args.adam_beta1, self.args.adam_beta2),
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=t_total * 0.1,
                                                    num_training_steps=t_total)

        # using mixed precision
        scaler = torch.cuda.amp.GradScaler()

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", self.args.epochs)
        logger.info("  Total optimization steps = %d", t_total)

        global_step, tr_loss, logging_loss = 0, 0.0, 0.0
        best_f1, best_global_step = -1, -1

        train_iterator = tqdm(range(int(self.args.epochs)), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_batches, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                batch['input_ids'] = torch.tensor(batch['input_ids'], device=self.device)
                batch['attention_mask'] = torch.tensor(batch['attention_mask'], device=self.device)
                batch['gold_clusters'] = torch.tensor(batch['gold_clusters'], device=self.device)
                if 'leftovers' in batch:
                    batch['leftovers']['input_ids'] = torch.tensor(batch['leftovers']['input_ids'], device=self.device)
                    batch['leftovers']['attention_mask'] = torch.tensor(batch['leftovers']['attention_mask'],
                                                                        device=self.device)

                self.model.zero_grad()
                self.model.train()

                with torch.cuda.amp.autocast():
                    outputs = self.model(batch, gold_clusters=batch['gold_clusters'], return_all_outputs=False)

                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                tr_loss += loss.item()
                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scheduler.step()  # Update learning rate schedule
                scaler.update()  # Updates the scale for next iteration
                global_step += 1

                # Log metrics
                if global_step % self.args.logging_steps == 0:
                    loss = (tr_loss - logging_loss) / self.args.logging_steps
                    logger.info(f"loss step {global_step}: {loss}")
                    self.wandb_logger({'loss': loss}, step=global_step)
                    logging_loss = tr_loss

                # Evaluation
                if self.dev_sampler is not None and global_step % self.args.eval_steps == 0:
                    results = self.evaluate(prefix=f'step_{global_step}', test=False)
                    self.wandb_logger(results, step=global_step)

                    f1 = results["f1"]
                    if f1 > best_f1:
                        best_f1, best_global_step = f1, global_step
                        self.wandb_runner.summary["best_f1"] = best_f1

                        # Save model
                        output_dir = os.path.join(self.args.output_dir, f'model')
                        save_all(tokenizer=self.tokenizer, model=self.model, output_dir=output_dir)
                    logger.info(f"best f1 is {best_f1} on global step {best_global_step}")

    def evaluate(self, test=False, prefix=''):
        if test:
            eval_sampler = self.test_sampler
            dataset_str = 'test_set'
        else:
            eval_sampler = self.dev_sampler
            dataset_str = 'dev_set'
        if eval_sampler is None:
            logger.info(f'Skipping evaluation. {dataset_str} is None')
            return {}

        self.model.eval()

        logger.info(f"***** Running evaluation on {dataset_str} - {len(eval_sampler.dataset)} documents *****")

        metrics_dict = {'loss': 0., 'post_pruning': MentionEvaluator(), 'mentions': MentionEvaluator(),
                        'coref': CorefEvaluator()}
        doc_to_tokens = {}
        doc_to_subtoken_map = {}
        doc_to_new_word_map = {}
        doc_to_prediction = {}

        with tqdm(desc="Inference", total=len(eval_sampler.dataset)) as progress_bar:
            for idx, batch in enumerate(eval_sampler):
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
