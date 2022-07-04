import os
import logging
import json
import spacy
from pathlib import Path
from tqdm import tqdm
import random
import pandas as pd
import torch
import numpy as np
from consts import NULL_ID_FOR_COREF


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


def to_dataframe(file_path):
    global nlp
    df = pd.read_json(file_path, lines=True)

    if 'tokens' in df.columns:
        pass
    elif 'sentences' in df.columns:
        # this is just for ontonotes. please avoid using 'sentences' and use 'text' or 'tokens'
        df['tokens'] = df['sentences'].apply(lambda x: flatten(x))
    elif 'text' in df.columns:
        if nlp is None:
            nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "lemmatizer", "ner", "textcat"])
        texts = df['text'].tolist()
        logger.info(f'Tokenize documents using Spacy...')
        df['tokens'] = [[tok.text for tok in doc] for doc in tqdm(nlp.pipe(texts), total=len(texts))]
    else:
        raise NotImplementedError(f'The jsonlines must include tokens/text/sentences attribute')

    if 'speakers' in df.columns:
        df['speakers'] = df['speakers'].apply(lambda x: flatten(x))
    else:
        df['speakers'] = df['tokens'].apply(lambda x: [None] * len(x))

    if 'doc_key' not in df.columns:
        raise NotImplementedError(f'The jsonlines must include doc_key, you can use uuid.uuid4().hex to generate.')

    if 'clusters' in df.columns:
        df = df[['doc_key', 'tokens', 'speakers', 'clusters']]
    else:
        df = df[['doc_key', 'tokens', 'speakers']]

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
