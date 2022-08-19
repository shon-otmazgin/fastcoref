import logging
import os.path
from collections import defaultdict

import datasets
from datasets.fingerprint import Hasher
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from fastcoref.utilities import util, consts
from fastcoref.utilities.collate import SegmentCollator, LongformerCollator

logger = logging.getLogger(__name__)


def _tokenize(tokenizer, tokens, clusters, speakers):
    token_to_new_token_map = []
    new_token_map = []
    new_tokens = []
    last_speaker = None

    for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
        if last_speaker != speaker:
            new_tokens += [consts.SPEAKER_START, speaker, consts.SPEAKER_END]
            new_token_map += [None, None, None]
            last_speaker = speaker
        token_to_new_token_map.append(len(new_tokens))
        new_token_map.append(idx)
        new_tokens.append(token)

    for cluster in clusters:
        for start, end in cluster:
            assert tokens[start:end + 1] == new_tokens[token_to_new_token_map[start]:token_to_new_token_map[end] + 1]

    encoded_text = tokenizer(new_tokens, add_special_tokens=True, is_split_into_words=True)

    new_clusters = [[(encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                      encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1)
                     for start, end in cluster] for cluster in clusters]

    return {'tokens': tokens,
            'input_ids': encoded_text['input_ids'],
            'gold_clusters': new_clusters,
            'subtoken_map': encoded_text.word_ids(),
            'new_token_map': new_token_map
            }


def encode(example, tokenizer):
    if 'clusters' not in example:
        example['clusters'] = []
    encoded_example = _tokenize(tokenizer, example['tokens'], example['clusters'], example['speakers'])

    gold_clusters = encoded_example['gold_clusters']
    encoded_example['num_clusters'] = len(gold_clusters) if gold_clusters else 0
    encoded_example['max_cluster_size'] = max(len(c) for c in gold_clusters) if gold_clusters else 0
    encoded_example['length'] = len(encoded_example['input_ids'])

    return encoded_example


def create(tokenizer, train_file=None, dev_file=None, test_file=None, cache_dir='cache', api=False):
    if train_file is None and dev_file is None and test_file is None:
        raise Exception(f'Provide at least train/dev/test file to create the dataset')

    dataset_files = {'train': train_file, 'dev': dev_file, 'test': test_file}

    cache_key = Hasher.hash(dataset_files)
    dataset_path = os.path.join(cache_dir, cache_key)

    try:
        dataset = datasets.load_from_disk(dataset_path)
        logger.info(f'Dataset restored from: {dataset_path}')
    except FileNotFoundError:
        logger.info(f'Creating dataset...')

        dataset_dict = {}
        for split, path in dataset_files.items():
            if path is not None:
                df = util.to_dataframe(path, api=api)
                dataset_dict[split] = Dataset.from_pandas(df)

        dataset = DatasetDict(dataset_dict)
        logger.info(f'Tokenize tokens with HuggingFace...')
        dataset = dataset.map(encode, batched=False, fn_kwargs={'tokenizer': tokenizer})
        dataset = dataset.remove_columns(column_names=['speakers', 'clusters'])

        logger.info(f'Saving dataset to: {dataset_path}')
        dataset.save_to_disk(dataset_path)

    return dataset, dataset_files


def create_batches(sampler, dataset_files, cache_dir='cache'):
    key = Hasher.hash(dataset_files)
    if isinstance(sampler.collator, SegmentCollator):
        key += '_segment_collator'
    elif isinstance(sampler.collator, LongformerCollator):
        key += '_longformer_collator'
    else:
        raise NotImplementedError('this collator not implemented!')

    cache_key = Hasher.hash(key)
    dataset_path = os.path.join(cache_dir, cache_key)

    try:
        batches = datasets.load_from_disk(dataset_path)
        logger.info(f'Batches restored from: {dataset_path}')
    except FileNotFoundError:
        logger.info(f'Creating batches for {len(sampler.dataset)} examples...')

        # huggingface dataset cannot save tensors. so we will save lists and on train loop transform to tensors.
        batches_dict = defaultdict(lambda: [])

        for i, batch in enumerate(tqdm(sampler)):
            for k, v in batch.items():
                batches_dict[k].append(v)

        batches = Dataset.from_dict(batches_dict)
        logger.info(f'{len(batches)} batches created.')

        logger.info(f'Saving batches to {dataset_path}')
        batches.save_to_disk(dataset_path)

    return batches
