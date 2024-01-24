# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphTemporalLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 n_heads,
                 dropout,
                 residual,
                 num_time_steps):
        super(GraphTemporalLayer, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.residual = residual
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.feedforward_linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.feedforward_linear_2 = nn.Linear(hidden_dim, 2)
        self.attention_dropout = nn.Dropout(dropout)
        self.num_time_steps = num_time_steps
        print('num_time_steps: ', self.num_time_steps)
        self.position_embedding_temporal = nn.Embedding(self.num_time_steps, hidden_dim)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.LSTM = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.init_weights()

    def forward(self, structural_output, position_embedding_clustering_coefficient,
                position_embedding_bidirectional_links_ratio, exist_nodes):
        structural_input = structural_output

        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(structural_output.shape[0], 1).long().to(structural_output.device)
        position_embedding_temporal = self.position_embedding_temporal(position_inputs)
        temporal_inputs = structural_output + position_embedding_temporal + position_embedding_clustering_coefficient + position_embedding_bidirectional_links_ratio
        temporal_inputs = self.layer_norm(temporal_inputs)
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)
        outputs = torch.matmul(q_, k_.permute(0, 2, 1))
        outputs = outputs / (split_size ** 0.5)
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        sequence_mask = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
        total_mask = sequence_mask
        total_mask = total_mask.float()
        padding = torch.ones_like(total_mask) * (-1e9)
        outputs = torch.where(total_mask == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        attention_weight = outputs.data
        outputs = self.attention_dropout(outputs)
        outputs = torch.matmul(outputs, v_)
        multi_head_attention_output = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),dim=2)  # [B, T, F]
        if self.residual:
            multi_head_attention_output += structural_input
        multi_head_attention_output = self.layer_norm(multi_head_attention_output)
        multi_head_attention_output, node_representation = self.feed_forward(multi_head_attention_output)
        return multi_head_attention_output, (node_representation, attention_weight)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.Q_embedding_weights)
        nn.init.kaiming_uniform_(self.K_embedding_weights)
        nn.init.kaiming_uniform_(self.V_embedding_weights)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()

    def feed_forward(self, inputs):
        out = self.feedforward_linear_1(inputs)
        out = self.activation(out)
        node_representation = out.data
        out = self.feedforward_linear_2(out)
        return out, node_representation

