import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
from transformers import BertPreTrainedModel, AutoModel
from transformers.activations import ACT2FN

from utilities.util import extract_clusters, extract_mentions_to_clusters, mask_tensor

# took from: https://github.com/yuvalkirstain/s2e-coref


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


class FastMention(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_span_length = args.max_span_length
        self.top_lambda = args.top_lambda
        self.ffnn_size = args.ffnn_size
        self.do_mlps = self.ffnn_size > 0
        self.ffnn_size = self.ffnn_size if self.do_mlps else config.hidden_size

        base_model = AutoModel.from_config(config)
        FastMention.base_model_prefix = base_model.base_model_prefix
        FastMention.config_class = base_model.config_class
        setattr(self, self.base_model_prefix, base_model)

        self.start_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None
        self.end_mention_mlp = FullyConnectedLayer(config, config.hidden_size, self.ffnn_size, args.dropout_prob) if self.do_mlps else None

        self.mention_start_classifier = Linear(self.ffnn_size, 1)
        self.mention_end_classifier = Linear(self.ffnn_size, 1)
        self.mention_s2e_classifier = Linear(self.ffnn_size, self.ffnn_size)

        self.init_weights()

    def num_parameters(self) -> tuple:
        def head_filter(x):
            return x[1].requires_grad and any(hp in x[0] for hp in ['coref', 'mention', 'antecedent'])

        head_params = filter(head_filter, self.named_parameters())
        head_params = sum(p.numel() for n, p in head_params)
        return super().num_parameters() - head_params, head_params

    def _get_span_mask(self, batch_size, k, max_k):
        """
        :param batch_size: int
        :param k: tensor of size [batch_size], with the required k for each example
        :param max_k: int
        :return: [batch_size, max_k] of zero-ones, where 1 stands for a valid span and 0 for a padded span
        """
        size = (batch_size, max_k)
        idx = torch.arange(max_k, device=self.device).unsqueeze(0).expand(size)
        len_expanded = k.unsqueeze(1).expand(size)
        return (idx < len_expanded).int()

    def _prune_topk_mentions(self, mention_logits, attention_mask):
        """
        :param mention_logits: Shape [batch_size, seq_length, seq_length]
        :param attention_mask: [batch_size, seq_length]
        :param top_lambda:
        :return:
        """
        batch_size, seq_length, _ = mention_logits.size()
        actual_seq_lengths = torch.sum(attention_mask, dim=-1)  # [batch_size]

        k = (actual_seq_lengths * self.top_lambda).int()  # [batch_size]
        max_k = int(torch.max(k))  # This is the k for the largest input in the batch, we will need to pad

        _, topk_1d_indices = torch.topk(mention_logits.view(batch_size, -1), dim=-1, k=max_k)  # [batch_size, max_k]

        span_mask = self._get_span_mask(batch_size, k, max_k)  # [batch_size, max_k]
        # drop the invalid indices and set them to the last index
        topk_1d_indices = (topk_1d_indices * span_mask) + (1 - span_mask) * ((seq_length ** 2) - 1)  # We take different k for each example
        # sorting for coref mention order
        sorted_topk_1d_indices, _ = torch.sort(topk_1d_indices, dim=-1)  # [batch_size, max_k]

        # gives the row index in 2D matrix
        topk_mention_start_ids = torch.div(sorted_topk_1d_indices, seq_length, rounding_mode='floor') # [batch_size, max_k]
        topk_mention_end_ids = sorted_topk_1d_indices % seq_length  # [batch_size, max_k]

        topk_mention_logits = mention_logits[torch.arange(batch_size).unsqueeze(-1).expand(batch_size, max_k),
                                             topk_mention_start_ids, topk_mention_end_ids]  # [batch_size, max_k]

        # this is antecedents scores - rows mentions, cols coref mentions
        topk_mention_logits = topk_mention_logits.unsqueeze(-1) + topk_mention_logits.unsqueeze(-2)  # [batch_size, max_k, max_k]

        return topk_mention_start_ids, topk_mention_end_ids, span_mask, topk_mention_logits

    def _get_mention_labels(self, mention_logits, all_clusters):
        batch_size, seq_len, _ = mention_logits.size()
        new_cluster_labels = torch.zeros((batch_size, seq_len, seq_len), device='cpu')
        all_clusters_cpu = all_clusters.cpu().numpy()
        for b, gold_clusters in enumerate(all_clusters_cpu):
            gold_clusters = extract_clusters(gold_clusters)
            mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
            gold_mentions = set(mention_to_gold_clusters.keys())
            for start, end in gold_mentions:
                new_cluster_labels[b, start, end] = 1
        new_cluster_labels = new_cluster_labels.to(self.device)
        return new_cluster_labels

    def _get_binary_cross_entropy_loss(self, mention_logits, mention_labels):
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(mention_logits, mention_labels)
        return loss

    def _get_mention_mask(self, mention_logits_or_weights):
        """
        Returns a tensor of size [batch_size, seq_length, seq_length] where valid spans
        (start <= end < start + max_span_length) are 1 and the rest are 0
        :param mention_logits_or_weights: Either the span mention logits or weights, size [batch_size, seq_length, seq_length]
        """
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

    def forward(self, batch, gold_clusters=None, return_all_outputs=False):
        sequence_output, attention_mask = self.forward_transformer(batch)

        # Compute representations
        start_mention_reps = self.start_mention_mlp(sequence_output) if self.do_mlps else sequence_output
        end_mention_reps = self.end_mention_mlp(sequence_output) if self.do_mlps else sequence_output

        # mention scores
        mention_logits = self._calc_mention_logits(start_mention_reps, end_mention_reps)

        # prune mentions
        mention_start_ids, mention_end_ids, span_mask, topk_mention_logits = self._prune_topk_mentions(mention_logits, attention_mask)

        if return_all_outputs:
            outputs = (mention_start_ids, mention_end_ids, mention_logits)
        else:
            outputs = tuple()

        if gold_clusters is not None:
            mention_labels = self._get_mention_labels(mention_logits, gold_clusters)
            loss = self._get_binary_cross_entropy_loss(mention_logits, mention_labels)
            outputs = (loss,) + outputs

        return outputs


