import os
import logging
import json
import spacy
from spacy.cli import download
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd
import torch
import numpy as np
from fastcoref.utilities.consts import NULL_ID_FOR_COREF, CATEGORIES, PRONOUNS_GROUPS

logger = logging.getLogger(__name__)
nlp = None


def output_evaluation_metrics(metrics_dict, output_dir, prefix):
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


def align_clusters(clusters, subtoken_maps, new_word_maps):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for start, end in cluster:
            try:
                start, end = subtoken_maps[start], subtoken_maps[end]
            except IndexError:
                # this is padding index
                continue
            if start is None or end is None:
                continue
            start, end = new_word_maps[start], new_word_maps[end]
            new_cluster.append([start, end])
        new_clusters.append(new_cluster)
    return new_clusters


def align_clusters_to_char_level(clusters, char_map):
    new_clusters = []
    for cluster in clusters:
        new_cluster = []
        for start, end in cluster:
            span_idx, span_char_level = char_map[(start, end)]
            if span_char_level is None:
                continue
            new_cluster.append(span_char_level)
        new_clusters.append(new_cluster)
    return new_clusters


def align_to_char_level(span_starts, span_ends, subtoken_maps, new_word_maps, tokens_to_start_char, tokens_to_end_char):
    char_map = {}
    reverse_char_map = {}
    for idx, (start, end) in enumerate(zip(span_starts, span_ends)):
        try:
            new_start, new_end = subtoken_maps[start], subtoken_maps[end]
        except IndexError:
            # this is padding index
            char_map[(start, end)] = None, None
            continue
        if new_start is None or new_end is None:
            char_map[(start, end)] = None, None
            continue
        new_start, new_end = new_word_maps[new_start], new_word_maps[new_end]
        new_start, new_end = tokens_to_start_char[new_start], tokens_to_end_char[new_end]
        char_map[(start, end)] = idx, (new_start, new_end)
        reverse_char_map[(new_start, new_end)] = idx, (start, end)

    return char_map, reverse_char_map


def flatten(l):
    return [item for sublist in l for item in sublist]


def read_jsonlines(file):
    with open(file, 'r') as f:
        docs = [json.loads(line.strip()) for line in f]
    return docs


def write_prediction_to_jsonlines(args, doc_to_prediction, doc_to_tokens, doc_to_subtoken_map, doc_to_new_word_map):
    eval_file = args.dataset_files[args.eval_split]
    if args.output_file is not None:
        output_eval_file = args.output_file
    else:
        output_eval_file = Path(eval_file).stem + '.output.jsonlines'
        if args.output_dir is not None:
            output_eval_file = os.path.join(args.output_dir, output_eval_file)
    logger.info(f'Predicted clusters at: {output_eval_file}')

    docs = read_jsonlines(file=eval_file)
    with open(output_eval_file, "w") as writer:
        for doc in docs:
            doc_key = doc['doc_key']
            assert doc_key in doc_to_prediction

            predicted_clusters = doc_to_prediction[doc_key]
            tokens = doc_to_tokens[doc_key]
            subtoken_map = doc_to_subtoken_map[doc_key]
            new_word_map = doc_to_new_word_map[doc_key]

            new_predicted_clusters = align_clusters(predicted_clusters, subtoken_map, new_word_map)
            doc['tokens'] = tokens
            doc['clusters'] = new_predicted_clusters

            writer.write(json.dumps(doc) + "\n")


def to_dataframe(file_path, api=False):
    global nlp
    df = pd.read_json(file_path, lines=True)

    if 'tokens' in df.columns:
        pass
    elif 'sentences' in df.columns:
        # this is just for ontonotes. please avoid using 'sentences' and use 'text' or 'tokens'
        df['tokens'] = df['sentences'].apply(lambda x: flatten(x))
    elif 'text' in df.columns:
        if nlp is None:
            try:
                nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
            except OSError:
                download('en_core_web_sm')
                nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        texts = df['text'].tolist()
        logger.info(f'Tokenize text with Spacy...')

        docs_tokens = []
        docs_tokens_to_start_char = []
        docs_tokens_to_end_char = []
        for doc in tqdm(nlp.pipe(texts), total=len(texts)):
            tokens = []
            tokens_to_start_char = []
            tokens_to_end_char = []
            for tok in doc:
                tokens.append(tok.text)
                tokens_to_start_char.append(tok.idx)
                tokens_to_end_char.append(tok.idx + len(tok.text))
            docs_tokens.append(tokens)
            docs_tokens_to_start_char.append(tokens_to_start_char)
            docs_tokens_to_end_char.append(tokens_to_end_char)

        df['tokens'] = docs_tokens
        df['tokens_to_start_char'] = docs_tokens_to_start_char
        df['tokens_to_end_char'] = docs_tokens_to_end_char
    else:
        raise NotImplementedError(f'The jsonlines must include tokens/text/sentences attribute')

    if 'speakers' in df.columns:
        df['speakers'] = df['speakers'].apply(lambda x: flatten(x))
    else:
        df['speakers'] = df['tokens'].apply(lambda x: [None] * len(x))

    if not api and 'doc_key' not in df.columns:
        raise NotImplementedError(f'The jsonlines must include doc_key, you can use uuid.uuid4().hex to generate.')

    columns = ['tokens', 'speakers']
    if not api:
        columns.append('doc_key')
    if 'text' in df.columns:
        columns.append('text')
        columns.append('tokens_to_start_char')
        columns.append('tokens_to_end_char')
    if 'clusters' in df.columns:
        columns.append('clusters')
    df = df[columns]

    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def save_all(model, tokenizer, output_dir):
    logger.info(f"Saving model to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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


def pad_clusters_inside(clusters, max_cluster_size):
    return [cluster + [(NULL_ID_FOR_COREF, NULL_ID_FOR_COREF)] * (max_cluster_size - len(cluster)) for cluster
            in clusters]


def pad_clusters_outside(clusters, max_num_clusters):
    return clusters + [[]] * (max_num_clusters - len(clusters))


def pad_clusters(clusters, max_num_clusters, max_cluster_size):
    clusters = pad_clusters_outside(clusters, max_num_clusters)
    clusters = pad_clusters_inside(clusters, max_cluster_size)
    return clusters


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


def softXEnt(teacher_logits, student_logits, span_mask, T=1):
    teacher_probs = torch.softmax(teacher_logits / T, dim=-1)
    student_log_probs = torch.log_softmax(student_logits / T, dim=-1)

    losses = (-teacher_probs * student_log_probs).sum(dim=-1)                # [batch_size, seq_len]

    losses = losses * span_mask
    per_example_loss = torch.sum(losses, dim=-1)                            # [batch_size]

    per_example_loss = per_example_loss / losses.size(-1)
    loss = per_example_loss.mean()

    return loss
