import numpy as np
from collections import Counter
from scipy.optimize import linear_sum_assignment

from consts import CATEGORIES


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)


class CorefCategories:
    def __init__(self):
        self.categories = {cat_name: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for cat_name in CATEGORIES}

    def update(self, coref_logits, categories_labels, clusters_labels):
        for cat_name, cat_id in CATEGORIES.items():
            cat_mask = categories_labels == cat_id

            labels = clusters_labels[cat_mask]
            logits = coref_logits[:, :, :-1][cat_mask]          # without null cluster logit

            self.categories[cat_name]['tn'] += np.logical_and(labels == 0., logits < 0).sum()
            self.categories[cat_name]['fn'] += np.logical_and(labels == 1., logits < 0).sum()
            self.categories[cat_name]['fp'] += np.logical_and(labels == 0., logits > 0).sum()
            self.categories[cat_name]['tp'] += np.logical_and(labels == 1., logits > 0).sum()

    def get_stats(self):
        stats = {}
        for cat_name in self.categories:
            tp, fp = self.categories[cat_name]['tp'], self.categories[cat_name]['fp']
            fn, tn = self.categories[cat_name]['fn'], self.categories[cat_name]['tn']

            stats[cat_name] = {'true_pairs': tp + fn, 'false_pairs': tn + fp, 'precision': 0, 'recall': 0, 'f1': 0}
            if tp + fp:
                stats[cat_name]['precision'] = round((tp / (tp + fp)) * 100, 1)
            if tp + fn:
                stats[cat_name]['recall'] = round((tp / (tp + fn)) * 100, 1)
            if tp + 0.5 * (fp + fn):
                stats[cat_name]['f1'] = round((tp / (tp + 0.5 * (fp + fn))) * 100, 1)
        return stats


class MentionEvaluator:
    def __init__(self):
        self.tp, self.fp, self.fn = 0, 0, 0

    def update(self, predicted_mentions, gold_mentions):
        predicted_mentions = set(predicted_mentions)
        gold_mentions = set(gold_mentions)

        self.tp += len(predicted_mentions & gold_mentions)
        self.fp += len(predicted_mentions - gold_mentions)
        self.fn += len(gold_mentions - predicted_mentions)

    def get_f1(self):
        pr = self.get_precision()
        rec = self.get_recall()
        return 2 * pr * rec / (pr + rec) if pr + rec > 0 else 0.0

    def get_recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    def get_precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()


class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = sum(scores[row_ind, col_ind])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem
