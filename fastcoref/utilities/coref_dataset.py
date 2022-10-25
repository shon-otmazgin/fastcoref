import json
import logging
import os.path
from collections import defaultdict

import datasets
from datasets import Dataset, Sequence, Value
from tqdm import tqdm

from fastcoref.utilities import util, consts

logger = logging.getLogger(__name__)


def add_speaker_information(tokens, speakers):
    token_to_new_token_map = []
    new_token_to_token_map = []
    new_tokens = []
    last_speaker = None

    for idx, (token, speaker) in enumerate(zip(tokens, speakers)):
        if last_speaker != speaker:
            new_tokens += [consts.SPEAKER_START, speaker, consts.SPEAKER_END]
            new_token_to_token_map += [None, None, None]
            last_speaker = speaker
        token_to_new_token_map.append(len(new_tokens))
        new_token_to_token_map.append(idx)
        new_tokens.append(token)

    return new_tokens, token_to_new_token_map, new_token_to_token_map


def _tokenize(tokenizer, tokens, clusters, speakers):
    new_tokens, token_to_new_token_map, new_token_to_token_map = tokens, list(range(len(tokens))), list(range(len(tokens)))
    if speakers:
        new_tokens, token_to_new_token_map, new_token_to_token_map = add_speaker_information(tokens, speakers)
        for cluster in clusters:
            for start, end in cluster:
                assert tokens[start:end + 1] == new_tokens[token_to_new_token_map[start]:token_to_new_token_map[end] + 1]

    encoded_text = tokenizer(
        new_tokens, add_special_tokens=True, is_split_into_words=True,
        return_length=True, return_attention_mask=False
    )

    # shifting clusters indices to align with bpe tokens
    # align clusters is the reason we can't do it in batches.
    new_clusters = [[(encoded_text.word_to_tokens(token_to_new_token_map[start]).start,
                      encoded_text.word_to_tokens(token_to_new_token_map[end]).end - 1)
                     for start, end in cluster] for cluster in clusters]

    return {'tokens': tokens,
            'input_ids': encoded_text['input_ids'],
            'length': encoded_text['length'][0],

            'gold_clusters': new_clusters,
            # tokens to tokens + speakers
            'new_token_map': new_token_to_token_map,
            # tokens + speakers to bpe
            'subtoken_map': encoded_text.word_ids(),
            }

# TODO: better to do it in batches
def encode(example, tokenizer, nlp):
    if 'tokens' in example and example['tokens']:
        pass
    elif 'text' in example and example['text']:
        clusters = example['clusters']
        spacy_doc = nlp(example['text'])
        example['tokens'] = [tok.text for tok in spacy_doc]

        new_clusters = [[(spacy_doc.char_span(start, end).start,
                          spacy_doc.char_span(start, end).end - 1)
                         for start, end in cluster] for cluster in clusters]
        # verify alignment
        for cluster, new_cluster in zip(clusters, new_clusters):
            for (s1, e1), (s2, e2) in zip(cluster, new_cluster):
                mention = [tok.text for tok in nlp(example['text'][s1:e1])]
                assert mention == example['tokens'][s2:e2 + 1], (mention, example['tokens'][s2:e2 + 1])

        example['clusters'] = new_clusters
    else:
        raise ValueError(f"Example is empty: {example}")

    encoded_example = _tokenize(tokenizer, example['tokens'], example['clusters'], example['speakers'])

    gold_clusters = encoded_example['gold_clusters']
    encoded_example['num_clusters'] = len(gold_clusters) if gold_clusters else 0
    encoded_example['max_cluster_size'] = max(len(c) for c in gold_clusters) if gold_clusters else 0

    return encoded_example


def create(file, tokenizer, nlp):
    def read_jsonlines(file):
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                doc = json.loads(line)
                if "text" not in doc and "tokens" not in doc and "sentences" not in doc:
                    raise ValueError(f'The jsonlines should contains at lt least "text", "sentences" or "tokens" field')

                minimum_doc = {}
                if "doc_key" not in doc:
                    minimum_doc["doc_key"] = str(i)
                else:
                    minimum_doc["doc_key"] = doc["doc_key"]

                if "text" not in doc:
                    minimum_doc["text"] = ""
                else:
                    minimum_doc["text"] = doc["text"]

                if "tokens" in doc:
                    minimum_doc["tokens"] = doc["tokens"]
                elif "sentences" in doc:
                    minimum_doc["tokens"] = util.flatten(doc["sentences"])
                else:
                    minimum_doc["tokens"] = []

                if "speakers" not in doc:
                    minimum_doc["speakers"] = []
                else:
                    minimum_doc["speakers"] = util.flatten(doc["speakers"])

                if "clusters" not in doc:
                    minimum_doc["clusters"] = []
                else:
                    minimum_doc["clusters"] = doc["clusters"]

                yield minimum_doc

    features = datasets.Features(
        {
            "doc_key": Value("string"),
            "text": Value("string"),
            "tokens": Sequence(Value("string")),
            "speakers": Sequence(Value("string")),
            "clusters": Sequence(Sequence(Sequence(Value("int64")))),
        }
    )

    dataset = Dataset.from_generator(read_jsonlines, features=features, gen_kwargs={'file': file})
    dataset = dataset.map(
        encode, batched=False,
        fn_kwargs={'tokenizer': tokenizer, 'nlp': nlp},
        load_from_cache_file=True
    )

    return dataset


# TODO: this function can be implemented much much better, e.g. from_generator
def create_batches(sampler, shuffle=True, cache_dir='cache'):
    # huggingface dataset cannot save tensors. so we will save lists and on train loop transform to tensors.
    batches_dict = defaultdict(lambda: [])

    for i, batch in enumerate(tqdm(sampler, desc="Creating batches for training")):
        for k, v in batch.items():
            batches_dict[k].append(v)

    batches = Dataset.from_dict(batches_dict)

    return batches