import logging
import numpy as np
import torch
import time

from utilities.metrics import CorefEvaluator, MentionEvaluator
from utilities.util import create_clusters, create_mention_to_antecedent, update_metrics, \
    output_evaluation_metrics, write_prediction_to_jsonlines
from tqdm.auto import tqdm

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

        metrics_dict = {'loss': 0., 'post_pruning': MentionEvaluator(), 'mentions': MentionEvaluator(),
                        'coref': CorefEvaluator()}
        doc_to_tokens = {}
        doc_to_subtoken_map = {}
        doc_to_new_word_map = {}
        doc_to_prediction = {}

        total_time = 0
        evaluation = False
        with tqdm(desc="Inference", total=len(self.eval_dataloader.dataset)) as progress_bar:
            for idx, batch in enumerate(self.eval_dataloader):
                doc_keys = batch['doc_key']
                tokens = batch['tokens']
                subtoken_map = batch['subtoken_map']
                new_token_map = batch['new_token_map']
                gold_clusters = batch['gold_clusters']

                start_time = time.time()
                with torch.no_grad():
                    outputs = model(batch, gold_clusters=gold_clusters, return_all_outputs=True)
                end_time = time.time()
                total_time += end_time - start_time

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                if gold_clusters is not None:
                    evaluation = True
                    gold_clusters = gold_clusters.cpu().numpy()
                    loss, span_starts, span_ends, mention_logits, coref_logits = outputs_np
                    metrics_dict['loss'] += loss.item()
                else:
                    span_starts, span_ends, mention_logits, coref_logits = outputs_np

                doc_indices, mention_to_antecedent = create_mention_to_antecedent(span_starts, span_ends, coref_logits)

                for i, doc_key in enumerate(doc_keys):
                    doc_mention_to_antecedent = mention_to_antecedent[np.nonzero(doc_indices == i)]
                    predicted_clusters = create_clusters(doc_mention_to_antecedent)

                    doc_to_prediction[doc_key] = predicted_clusters
                    doc_to_tokens[doc_key] = tokens[i]
                    doc_to_subtoken_map[doc_key] = subtoken_map[i]
                    doc_to_new_word_map[doc_key] = new_token_map[i]

                    if gold_clusters is not None:
                        update_metrics(metrics_dict, span_starts[i], span_ends[i], gold_clusters[i], predicted_clusters)

                progress_bar.update(n=len(doc_keys))

        write_prediction_to_jsonlines(
            self.args, doc_to_prediction,
            doc_to_tokens, doc_to_subtoken_map, doc_to_new_word_map
        )

        results = {}
        if evaluation:
            results = output_evaluation_metrics(
                metrics_dict=metrics_dict, output_dir=self.output_dir, prefix=prefix
            )
        logger.info(f'Total time: {total_time:.6f} seconds')

        return results
