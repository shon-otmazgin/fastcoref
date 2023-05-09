import logging
import os
import random
import torch
import numpy as np
from fastcoref.utilities.consts import NULL_ID_FOR_COREF, CATEGORIES, PRONOUNS_GROUPS

logger = logging.getLogger(__name__)


def flatten(l):
    return [item for sublist in l for item in sublist]


def save_all(model, tokenizer, output_dir):
    logger.info(f"Saving model to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def pad_clusters_inside(clusters, max_cluster_size):
    return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster)) for cluster
            in clusters]


def pad_clusters_outside(clusters, max_num_clusters):
    return clusters + [[]] * (max_num_clusters - len(clusters))


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters


def output_evaluation_metrics(metrics_dict, prefix):
    loss = metrics_dict['loss']
    post_pruning_mention_pr, post_pruning_mentions_r, post_pruning_mention_f1 = metrics_dict['post_pruning'].get_prf()
    mention_p, mentions_r, mention_f1 = metrics_dict['mentions'].get_prf()
    p, r, f1 = metrics_dict['coref'].get_prf()
    results = {
        'eval_loss': loss,
        "post pruning mention precision": post_pruning_mention_pr,
        "post pruning mention recall": post_pruning_mentions_r,
        "post pruning mention f1": post_pruning_mention_f1,
        "mention precision": mention_p,
        "mention recall": mentions_r,
        "mention f1": mention_f1,
        "precision": p,
        "recall": r,
        "f1": f1
    }

    logger.info("***** Eval results {} *****".format(prefix))
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"  {key : <30} = {value:.3f}")
        elif isinstance(value, dict):
            logger.info(f"  {key : <30} = {value}")

    return results



def update_metrics(metrics, span_starts, span_ends, gold_clusters, predicted_clusters):
    gold_clusters = extract_clusters(gold_clusters)
    candidate_mentions = list(zip(span_starts, span_ends))

    mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
    mention_to_predicted_clusters = extract_mentions_to_clusters(predicted_clusters)

    gold_mentions = list(mention_to_gold_clusters.keys())
    predicted_mentions = list(mention_to_predicted_clusters.keys())

    metrics['post_pruning'].update(candidate_mentions, gold_mentions)
    metrics['mentions'].update(predicted_mentions, gold_mentions)
    metrics['coref'].update(predicted_clusters, gold_clusters,
                            mention_to_predicted_clusters, mention_to_gold_clusters)


def encode(batch, tokenizer, nlp):
    if nlp is not None:
        tokenized_texts = tokenize_with_spacy(batch['text'], nlp)
    else:
        tokenized_texts = batch
        tokenized_texts['offset_mapping'] = [(list(zip(range(len(tokens)), range(1, 1 + len(tokens)))))
                                             for tokens in tokenized_texts['tokens']]
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

    # Edge case - Also disable other custom components
    all_pipe_names = nlp.pipe_names
    tokenizer_pipe_names = ["tok2vec"]

    disabled_pipe_names = [pipe_name for pipe_name in all_pipe_names if pipe_name not in tokenizer_pipe_names]
    docs = nlp.pipe(texts, disable=disabled_pipe_names)
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
                    # this is a special token index
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
