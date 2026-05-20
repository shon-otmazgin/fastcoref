import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, AutoModel
from transformers.activations import ACT2FN

from fastcoref.utilities.util import extract_clusters, extract_mentions_to_clusters, mask_tensor

# took from: https://github.com/yuvalkirstain/s2e-coref


def _is_head_param(named_param):
    return named_param[1].requires_grad and any(hp in named_param[0] for hp in ['coref', 'mention', 'antecedent'])


class FullyConnectedLayer(Module):
    def __init__(self, config, input_dim, output_dim, dropout_prob):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob

        self.dense = Linear(self.input_dim, self.output_dim)
        self.layer_norm = LayerNorm(self.output_dim, eps=config.layer_norm_eps)
        self.activation_func = ACT2FN[config.hidden_act]
        self.dropout = Dropout(self.dropout_prob)

    def forward(self, inputs):
        temp = inputs
        temp = self.dense(temp)
        temp = self.activation_func(temp)
        temp = self.layer_norm(temp)
        temp = self.dropout(temp)
        return temp


class FCorefModel(BertPreTrainedModel):
    all_tied_weights_keys: dict = {}

    def __init__(self, config):
        super().__init__(config)
        self.max_span_length = config.coref_head['max_span_length']
        self.top_lambda = config.coref_head['top_lambda']
        self.ffnn_size = config.coref_head['ffnn_size']
        self.dropout_prob = config.coref_head['dropout_prob']

        # RoBERTa/BERT support SDPA for 2-4x attention speedup
        base_model = AutoModel.from_config(config, attn_implementation="sdpa")
        FCorefModel.base_model_prefix = base_model.base_model_prefix
        FCorefModel.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, self.dropout_prob)
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, self.dropout_prob)

        self.start_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, self.dropout_prob)
        self.end_coref_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, self.dropout_prob)

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.antecedent_s2s_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)
        self.antecedent_e2s_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.init_weights()

    def num_parameters(self) -> tuple:
        head_params = filter(_is_head_param, self.named_parameters())
        head_params = sum(p.numel() for n, p in head_params)
        return super().num_parameters() - head_params, head_params

    def _get_span_mask(self, batch_size, k, max_k):
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask, topk_1d_indices):
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        if topk_1d_indices is None:
            _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]

        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        # drop the invalid indices and set them to the last index
        seq_length_t = torch.tensor(seq_length, device=mention_logits.device)
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * (seq_length_t * seq_length_t - 1)  # We take different k for each example
        # sorting for coref mention order
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        # gives the row index in 2D matrix
        topk_mention_start_ids = sorted_topk_1d_indices.div(seq_length_t, rounding_mode='floor')  # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices.remainder(seq_length_t)  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        # this is antecedents scores - rows mentions, cols coref mentions
        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _mask_antecedent_logits(self, antecedent_logits, span_mask):
        # We now build the matrix for each pair of spans (i,j) - whether j is a candidate for being antecedent of i?
        antecedents_mask = torch.ones_like(antecedent_logits, dtype=self.dtype).tril(diagonal=-1)  # [batch_size, k, k]
        antecedents_mask = antecedents_mask * span_mask.unsqueeze(-1)  # [batch_size, k, k]
        antecedent_logits = mask_tensor(antecedent_logits, antecedents_mask)
        return antecedent_logits

    def _get_cluster_labels_after_pruning(self, span_starts, span_ends, all_clusters):
        batch_size, max_k = span_starts.size()
        new_cluster_labels = torch.zeros((batch_size, max_k, max_k + 1), device='cpu')

        span_starts_cpu = span_starts.cpu()
        span_ends_cpu = span_ends.cpu()
        all_clusters_cpu = all_clusters.cpu().numpy()

        for b in range(batch_size):
            gold_clusters = extract_clusters(all_clusters_cpu[b])
            if not gold_clusters:
                continue
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())

            starts = span_starts_cpu[b].tolist()
            ends = span_ends_cpu[b].tolist()

            # Build a mapping from mention -> index for fast lookup
            mention_to_idx = {}
            for i, (s, e) in enumerate(zip(starts, ends)):
                if (s, e) in gold_mentions:
                    mention_to_idx.setdefault((s, e), []).append(i)

            # For each gold cluster, find all pruned mentions that belong to it
            # and mark all pairs as coreferent
            for cluster in gold_clusters:
                cluster_indices = []
                for mention in cluster:
                    if mention in mention_to_idx:
                        cluster_indices.extend(mention_to_idx[mention])
                # Mark pairs: for each mention i, all earlier mentions j in same cluster are antecedents
                for idx_pos, i in enumerate(cluster_indices):
                    for j in cluster_indices[:idx_pos]:
                        if j < i:
                            new_cluster_labels[b, i, j] = 1

        new_cluster_labels = new_cluster_labels.to(self.device)
        no_antecedents = 1 - torch.sum(new_cluster_labels, dim=-1).bool().float()
        new_cluster_labels[:, :, -1] = no_antecedents
        return new_cluster_labels

    def _get_marginal_log_likelihood_loss(self, coref_logits, cluster_labels_after_pruning, span_mask):
        gold_coref_logits = mask_tensor(coref_logits, cluster_labels_after_pruning)

        gold_log_sum_exp = torch.logsumexp(gold_coref_logits, dim=-1)  # [batch_size, max_k]
        all_log_sum_exp = torch.logsumexp(coref_logits, dim=-1)  # [batch_size, max_k]

        gold_log_probs = gold_log_sum_exp - all_log_sum_exp
        losses = - gold_log_probs

        losses = losses * span_mask
        per_example_loss = torch.sum(losses, dim=-1)  # [batch_size]

        per_example_loss = per_example_loss / losses.size(-1)
        loss = per_example_loss.mean()
        return loss

    def _get_mention_mask(self, mention_logits_or_weights):
        mention_mask = torch.ones_like(mention_logits_or_weights, dtype=self.dtype)
        mention_mask = mention_mask.triu(diagonal=0)
        mention_mask = mention_mask.tril(diagonal=self.max_span_length - 1)
        return mention_mask

    def _calc_mention_logits(self, start_mention_reps, end_mention_reps):
        start_mention_logits = self.mention_start_classifier(start_mention_reps).squeeze(-1)  # [batch_size, seq_length]
        end_mention_logits = self.mention_end_classifier(end_mention_reps).squeeze(-1)  # [batch_size, seq_length]

        temp = self.mention_s2e_classifier(start_mention_reps)  # [batch_size, seq_length]
        joint_mention_logits = torch.matmul(temp,
                                            end_mention_reps.permute([0, 2, 1]))  # [batch_size, seq_length, seq_length]

        mention_logits = joint_mention_logits + start_mention_logits.unsqueeze(-1) + end_mention_logits.unsqueeze(-2)
        mention_mask = self._get_mention_mask(mention_logits)  # [batch_size, seq_length, seq_length]
        mention_logits = mask_tensor(mention_logits, mention_mask)  # [batch_size, seq_length, seq_length]
        return mention_logits

    def _calc_coref_logits(self, top_k_start_coref_reps, top_k_end_coref_reps):
        # s2s
        temp = self.antecedent_s2s_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2e
        temp = self.antecedent_e2e_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # s2e
        temp = self.antecedent_s2e_classifier(top_k_start_coref_reps)  # [batch_size, max_k, dim]
        top_k_s2e_coref_logits = torch.matmul(temp,
                                              top_k_end_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # e2s
        temp = self.antecedent_e2s_classifier(top_k_end_coref_reps)  # [batch_size, max_k, dim]
        top_k_e2s_coref_logits = torch.matmul(temp,
                                              top_k_start_coref_reps.permute([0, 2, 1]))  # [batch_size, max_k, max_k]

        # sum all terms
        coref_logits = top_k_s2e_coref_logits + top_k_e2s_coref_logits + top_k_s2s_coref_logits + top_k_e2e_coref_logits  # [batch_size, max_k, max_k]
        return coref_logits

    def forward_transformer(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        docs, segments, segment_len = input_ids.size()
        input_ids, attention_mask = input_ids.view(-1, segment_len), attention_mask.view(-1, segment_len)

        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        attention_mask = attention_mask.view((docs, segments * segment_len))        # [docs, seq_len]
        sequence_output = sequence_output.view((docs, segments * segment_len, -1))  # [docs, seq_len, dim]

        leftovers_ids, leftovers_mask = batch['leftovers']['input_ids'], batch['leftovers']['attention_mask']
        if len(leftovers_ids) > 0:
            res_outputs = self.base_model(leftovers_ids, attention_mask=leftovers_mask)
            res_sequence_output = res_outputs.last_hidden_state

            attention_mask = torch.cat([attention_mask, leftovers_mask], dim=1)
            sequence_output = torch.cat([sequence_output, res_sequence_output], dim=1)

        return sequence_output, attention_mask

    def forward(self, batch, gold_clusters=None, topk_1d_indices=None, return_all_outputs=False):
        sequence_output, attention_mask = self.forward_transformer(batch)

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output)
        end_mention_reps = self.end_mention_mlp(sequence_output)

        start_coref_reps = self.start_coref_mlp(sequence_output)
        end_coref_reps = self.end_coref_mlp(sequence_output)

        # mention scores
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, attention_mask, topk_1d_indices)

        batch_size, _, dim = start_coref_reps.size()
        max_k = mention_start_ids.size(-1)
        size = (batch_size, max_k, dim)

        # Antecedent scores
        # gather reps
        topk_start_coref_reps = torch.gather(start_coref_reps, dim=1, index=mention_start_ids.unsqueeze(-1).expand(size))
        topk_end_coref_reps = torch.gather(end_coref_reps, dim=1, index=mention_end_ids.unsqueeze(-1).expand(size))
        coref_logits = self._calc_coref_logits(topk_start_coref_reps, topk_end_coref_reps)

        final_logits = topk_mention_logits + coref_logits
        final_logits = self._mask_antecedent_logits(final_logits, span_mask)
        # adding zero logits for null span
        final_logits = torch.cat((final_logits, torch.zeros((batch_size, max_k, 1), device=self.device)), dim=-1)  # [batch_size, max_k, max_k + 1]

        if return_all_outputs:
            outputs = (mention_start_ids, mention_end_ids, mention_logits, final_logits)
        else:
            outputs = tuple()

        if topk_1d_indices is not None:
            outputs = (span_mask,) + outputs

        if gold_clusters is not None:
            labels_after_pruning = self._get_cluster_labels_after_pruning(mention_start_ids, mention_end_ids, gold_clusters)
            loss = self._get_marginal_log_likelihood_loss(final_logits, labels_after_pruning, span_mask)
            outputs = (loss, ) + outputs

        return outputs
