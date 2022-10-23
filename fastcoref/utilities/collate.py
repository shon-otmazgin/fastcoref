import logging
import torch
import math
from fastcoref.utilities.util import pad_clusters


logger = logging.getLogger(__name__)


class LeftOversCollator:
    def __init__(self, tokenizer, device, max_segment_len):
        self.tokenizer = tokenizer
        self.device = device
        self.max_segment_len = max_segment_len

    def __call__(self, batch):
        # pad to the longest doc in the batch
        batch = self.tokenizer.pad(batch)
        batch['leftovers'] = {'input_ids': [], 'attention_mask': []}

        # break down to segment of segment len
        input_ids = [[ids[i:i + self.max_segment_len] for i in range(0, len(ids), self.max_segment_len)]
                     for ids in batch['input_ids']]
        attention_mask = [[mask[i:i + self.max_segment_len] for i in range(0, len(mask), self.max_segment_len)]
                          for mask in batch['attention_mask']]

        # if we have more than 1 segment and the last segment is less than segment_len we have leftovers.
        if len(input_ids[0]) > 1 and len(input_ids[0][-1]) < self.max_segment_len:
            batch['leftovers']['input_ids'] = torch.tensor([ids[-1] for ids in input_ids], device=self.device)
            batch['leftovers']['attention_mask'] = torch.tensor([mask[-1] for mask in attention_mask], device=self.device)

            # remove leftovers from main batch
            input_ids = [ids[:-1] for ids in input_ids]
            attention_mask = [mask[:-1] for mask in attention_mask]

        batch['input_ids'] = torch.tensor(input_ids, device=self.device)
        batch['attention_mask'] = torch.tensor(attention_mask, device=self.device)

        if 'gold_clusters' in batch:
            max_num_clusters, max_max_cluster_size = max(batch['num_clusters']), max(batch['max_cluster_size'])
            if max_num_clusters and max_max_cluster_size:
                padded_clusters = [pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in
                                   batch['gold_clusters']]
                batch['gold_clusters'] = torch.tensor(padded_clusters, device=self.device)
            else:
                batch['gold_clusters'] = None

        return batch


class PadCollator:
    def __init__(self, tokenizer, device, max_segment_len=512):
        self.tokenizer = tokenizer
        self.device = device
        self.max_segment_len = max_segment_len

    def __call__(self, batch):
        # pad to the longest doc in the batch
        batch = self.tokenizer.pad(batch)

        batch['input_ids'] = torch.tensor(batch['input_ids'], device=self.device)
        batch['attention_mask'] = torch.tensor(batch['attention_mask'], device=self.device)

        if 'gold_clusters' in batch:
            max_num_clusters, max_max_cluster_size = max(batch['num_clusters']), max(batch['max_cluster_size'])
            if max_num_clusters and max_max_cluster_size:
                padded_clusters = [pad_clusters(cluster, max_num_clusters, max_max_cluster_size) for cluster in
                                   batch['gold_clusters']]
                batch['gold_clusters'] = torch.tensor(padded_clusters, device=self.device)
            else:
                batch['gold_clusters'] = None

        return batch


class DynamicBatchSampler:
    def __init__(self, dataset, collator, max_tokens, max_segment_len, max_doc_len=None):
        self.max_tokens = max_tokens
        self.dataset = dataset.sort('length', reverse=False)
        self.collator = collator
        self.max_segment_len = max_segment_len
        self.max_doc_len = max_doc_len

    def __iter__(self):
        batch = []
        per_example_batch_len = 0
        for example in self.dataset:
            if self.max_doc_len is not None and example['length'] > self.max_doc_len:
                logger.info(f'Skipping doc with len {example["length"]}. max_doc_len is {self.max_doc_len}')
                continue
            if not batch:
                per_example_batch_len = self.calc_effective_per_example_batch_len(example['length'])
            elif (len(batch) + 1) * per_example_batch_len > self.max_tokens:
                yield self.collator(batch)
                batch = []
                per_example_batch_len = self.calc_effective_per_example_batch_len(example['length'])
            batch.append(example)
        if len(batch) > 0:
            yield self.collator(batch)

    def calc_effective_per_example_batch_len(self, example_len):
        return math.ceil(example_len / self.max_segment_len) * self.max_segment_len
