import logging
import numpy as np
import torch

from consts import CATEGORIES, STOPWORDS
from metrics import CorefEvaluator, MentionEvaluator, CorefCategories
from util import create_clusters, create_mention_to_antecedent, update_metrics, \
    output_evaluation_metrics, write_prediction_to_jsonlines
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, args, eval_dataloader):
        self.args = args
        self.output_dir = args.output_dir
        self.eval_dataloader = eval_dataloader

    def evaluate(self, model, prefix=""):
        # Eval!
        model.eval()

        logger.info(f"***** Running Inference on {self.args.eval_split} split {prefix} *****")
        logger.info(f"  Examples number: {len(self.eval_dataloader.dataset)}")

        pairs_examples = {cat_name: [] for cat_name in CATEGORIES}

        with tqdm(desc="Inference", total=len(self.eval_dataloader.dataset)) as progress_bar:
            for idx, batch in enumerate(self.eval_dataloader):
                doc_keys = batch['doc_key']
                tokens = batch['tokens']
                subtoken_map = batch['subtoken_map']
                new_token_map = batch['new_token_map']
                gold_clusters = batch['gold_clusters']

                with torch.no_grad():
                    outputs = model(batch, gold_clusters=gold_clusters, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                loss, span_starts, span_ends, coref_logits, categories_labels, clusters_labels = outputs_np

                type = 0
                for cat_name, cat_id in CATEGORIES.items():
                    doc_indices, span_indices, ant_indices = ((clusters_labels == type) * (categories_labels == cat_id)).nonzero()

                    for i, q, c in list(zip(doc_indices, span_indices, ant_indices))[:500]:
                        antecedent = int(span_starts[i, c]), int(span_ends[i, c])
                        mention = int(span_starts[i, q]), int(span_ends[i, q])

                        token_indices = [new_token_map[i][idx] for idx in subtoken_map[i][antecedent[0]:antecedent[1] + 1] if idx is not None]
                        token_indices = [idx for idx in token_indices if idx is not None]
                        if not token_indices:
                            continue
                        sentence_start = max(0, token_indices[0] - 10)
                        antecedent = [tokens[i][idx] for idx in token_indices if idx is not None]

                        token_indices = [new_token_map[i][idx] for idx in subtoken_map[i][mention[0]:mention[1] + 1] if idx is not None]
                        token_indices = [idx for idx in token_indices if idx is not None]
                        if not token_indices:
                            continue
                        sentence_end = min(token_indices[-1] + 10, len(tokens[i]))
                        mention = [tokens[i][idx] for idx in token_indices if idx is not None]

                        if sentence_end - sentence_start > 30:
                            continue

                        sentence = ' '.join(tokens[i][sentence_start:sentence_end+1])

                        pairs_examples[cat_name].append((mention, antecedent, sentence))

                progress_bar.update(n=len(doc_keys))

        for cat_name, pairs in pairs_examples.items():
            with open(f'/home/nlp/shon711/WWW/{"neg" if type == 0 else "pos"}/{cat_name}.txt', 'w') as f:
                for antecedent, mention, sentence in pairs:
                    f.write(f'{" ".join(antecedent)}\n{" ".join(mention)}\n{sentence}\n')
                    f.write(f'\n--------------------------------\n\n')

        return {}
