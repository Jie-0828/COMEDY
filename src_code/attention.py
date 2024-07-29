import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_w = nn.Parameter(torch.FloatTensor(self.in_channels, self.out_channels))
        self.weight_a = nn.Parameter(torch.FloatTensor(self.out_channels*2, 1))
        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight_w)
        nn.init.xavier_uniform_(self.weight_a)

    def forward(self, x_i,x_j):
        wh1 = torch.mm(x_i.unsqueeze(dim=0), self.weight_w)
        wh2 = torch.mm(x_j.unsqueeze(dim=0), self.weight_w)
        x_cat = torch.cat([wh1, wh2], dim=1)
        e = torch.mm(x_cat, self.weight_a)

        return e




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        self.linear = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.WQ(Q)
        K = self.WK(K)
        V = self.WV(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.combine_heads(output)
        output = self.linear(output)

        return output, attention_weights

class NodeFeatureAggregator(nn.Module):
    def __init__(self, d_model, num_heads):
        super(NodeFeatureAggregator, self).__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
    def forward(self, node_features, mask=None):
        if len(node_features)==1:
            node_features = node_features.unsqueeze(0)
        else:
            node_features = node_features[1:].unsqueeze(0)
        output, _ = self.attention(node_features, node_features, node_features, mask)

        output = output.sum(dim=1)
        # output=self.fc1(output)
        # output=self.relu(output)

        return output




