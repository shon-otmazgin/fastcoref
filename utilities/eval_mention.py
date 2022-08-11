import logging
import torch

from metrics import MentionEvaluator
from tqdm.auto import tqdm

from utilities.util import extract_mentions_to_clusters, extract_clusters

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

        metrics_dict = {'loss': 0., 'post_pruning': MentionEvaluator()}

        with tqdm(desc="Inference", total=len(self.eval_dataloader.dataset)) as progress_bar:
            for idx, batch in enumerate(self.eval_dataloader):
                doc_keys = batch['doc_key']
                gold_clusters = batch['gold_clusters']

                with torch.no_grad():
                    outputs = model(batch, gold_clusters=gold_clusters, return_all_outputs=True)

                outputs_np = tuple(tensor.cpu().numpy() for tensor in outputs)

                gold_clusters = gold_clusters.cpu().numpy()
                loss, span_starts, span_ends, mention_logits = outputs_np
                metrics_dict['loss'] += loss.item()

                for i, doc_key in enumerate(doc_keys):
                    gold_clusters_i = extract_clusters(gold_clusters[i])
                    mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters_i)
                    gold_mentions = set(mention_to_gold_clusters.keys())

                    candidate_mentions = list(zip(span_starts[i], span_ends[i]))
                    metrics_dict['post_pruning'].update(candidate_mentions, gold_mentions)

                progress_bar.update(n=len(doc_keys))

        post_pruning_mention_pr, post_pruning_mentions_r, post_pruning_mention_f1 = metrics_dict['post_pruning'].get_prf()
        results = {
            'eval_loss': metrics_dict['loss'],
            "precision": post_pruning_mention_pr,
            "recall": post_pruning_mentions_r,
            "f1": post_pruning_mention_f1,
        }
        logger.info("***** Eval results {} *****".format(prefix))
        for key, value in results.items():
            if isinstance(value, float):
                logger.info(f"  {key : <30} = {value:.3f}")
            elif isinstance(value, dict):
                logger.info(f"  {key : <30} = {value}")

        return results
