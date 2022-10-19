import logging
from tqdm import tqdm
import random
import torch
import numpy as np
from fastcoref.utilities.consts import NULL_ID_FOR_COREF, CATEGORIES, PRONOUNS_GROUPS

logger = logging.getLogger(__name__)


def encode(batch, tokenizer, nlp):
    tokenized_texts = tokenize_with_spacy(batch['text'], nlp)
    encoded_batch = tokenizer(
        tokenized_texts['tokens'], add_special_tokens=True, is_split_into_words=True,
        return_length=True, return_attention_mask=False
    )
    return {
        'tokens': tokenized_texts['tokens'],
        'input_ids': encoded_batch['input_ids'],
        'length': encoded_batch['length'],

        # bpe token -> spacy tokens
        'subtoken_map': [enc.word_ids for enc in encoded_batch.encodings],
        # this is a can use for speaker info TODO: better name!
        'new_token_map': [list(range(len(tokens))) for tokens in tokenized_texts['tokens']],
        # spacy tokens -> text char
        'offset_mapping': tokenized_texts['offset_mapping']
    }


def tokenize_with_spacy(texts, nlp):
    def handle_doc(doc):
        tokens = []
        offset_mapping = []
        for tok in doc:
            tokens.append(tok.text)
            offset_mapping.append((tok.idx, tok.idx + len(tok.text)))
        return tokens, offset_mapping

    tokenized_texts = {'tokens': [], 'offset_mapping': []}
    docs = nlp.pipe(texts)
    for doc in docs:
        tokens, offset_mapping = handle_doc(doc)
        tokenized_texts['tokens'].append(tokens)
        tokenized_texts['offset_mapping'].append(offset_mapping)

    return tokenized_texts


def align_to_char_level(span_starts, span_ends, token_to_char, subtoken_map=None, new_token_map=None):
    char_map = {}
    reverse_char_map = {}
    for idx, (start, end) in enumerate(zip(span_starts, span_ends)):
        new_start, new_end = start.copy(), end.copy()

        try:
            if subtoken_map is not None:
                new_start, new_end = subtoken_map[new_start], subtoken_map[new_end]
                if new_start is None or new_end is None:
                    char_map[(start, end)] = None, None
                    continue
            if new_token_map is not None:
                new_start, new_end = new_token_map[new_start], new_token_map[new_end]
            new_start, new_end = token_to_char[new_start][0], token_to_char[new_end][1]
            char_map[(start, end)] = idx, (new_start, new_end)
            reverse_char_map[(new_start, new_end)] = idx, (start, end)
        except IndexError:
            # this is padding index
            char_map[(start, end)] = None, None
            continue

    return char_map, reverse_char_map


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in cluster if NULL_ID_FOR_COREF not in m) for cluster in gold_clusters]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters


def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold


def create_clusters(mention_to_antecedent):
    # Note: mention_to_antecedent is a numpy array

    clusters, mention_to_cluster = [], {}
    for mention, antecedent in mention_to_antecedent:
        mention, antecedent = tuple(mention), tuple(antecedent)
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            if mention not in clusters[cluster_idx]:
                clusters[cluster_idx].append(mention)
                mention_to_cluster[mention] = cluster_idx
        elif mention in mention_to_cluster:
            cluster_idx = mention_to_cluster[mention]
            if antecedent not in clusters[cluster_idx]:
                clusters[cluster_idx].append(antecedent)
                mention_to_cluster[antecedent] = cluster_idx
        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])

    clusters = [tuple(cluster) for cluster in clusters]
    return clusters


def create_mention_to_antecedent(span_starts, span_ends, coref_logits):
    batch_size, n_spans, _ = coref_logits.shape

    max_antecedents = coref_logits.argmax(axis=-1)
    doc_indices, mention_indices = np.nonzero(max_antecedents < n_spans)        # indices where antecedent is not null.
    antecedent_indices = max_antecedents[max_antecedents < n_spans]
    span_indices = np.stack([span_starts, span_ends], axis=-1)

    mentions = span_indices[doc_indices, mention_indices]
    antecedents = span_indices[doc_indices, antecedent_indices]
    mention_to_antecedent = np.stack([mentions, antecedents], axis=1)

    return doc_indices, mention_to_antecedent


def mask_tensor(t, mask):
    t = t + ((1.0 - mask.float()) * -10000.0)
    t = torch.clamp(t, min=-10000.0, max=10000.0)
    return t


def get_pronoun_id(span):
    if len(span) == 1:
        span = list(span)
        if span[0] in PRONOUNS_GROUPS:
            return PRONOUNS_GROUPS[span[0]]
    return -1


def get_category_id(mention, antecedent):
    mention, mention_pronoun_id = mention
    antecedent, antecedent_pronoun_id = antecedent

    if mention_pronoun_id > -1 and antecedent_pronoun_id > -1:
        if mention_pronoun_id == antecedent_pronoun_id:
            return CATEGORIES['pron-pron-comp']
        else:
            return CATEGORIES['pron-pron-no-comp']

    if mention_pronoun_id > -1 or antecedent_pronoun_id > -1:
        return CATEGORIES['pron-ent']

    if mention == antecedent:
        return CATEGORIES['match']

    union = mention.union(antecedent)
    if len(union) == max(len(mention), len(antecedent)):
        return CATEGORIES['contain']

    return CATEGORIES['other']
