'''
Transformer implementation with PyTorch
'''

import torch
import math

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int = 8, num_heads: int = 2):
        super.__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads

        self.WQ = torch.nn.Linear(d_model, d_model)
        self.WK = torch.nn.Linear(d_model, d_model)
        self.WV = torch.nn.Linear(d_model, d_model)
        self.output_project = torch.nn.Linear(d_model, d_model)

    def forward(self, X):
        Q = self.WQ(X)
        K = self.WK(X)
        V = self.WV(X)

        Q_heads = torch.split(Q, self.head_dim, dim=-1)
        K_heads = torch.split(K, self.head_dim, dim=-1)
        V_heads = torch.split(V, self.head_dim, dim=-1)
        # process heads
        scores_head0 = Q_heads[0] @ K_heads[0].T
        scores_head0 /= math.sqrt(self.head_dim)

        scores_head1 = Q_heads[1] @ K_heads[1].T
        scores_head1 /= math.sqrt(self.head_dim)
        # causal mask
        mask_head0 = torch.triu(torch.ones(scores_head0.shape, dtype=torch.bool), diagonal=1)
        scores_head0 = scores_head0.mask_fill(mask_head0, float("-inf"))
        mask_head1 = torch.triu(torch.ones(scores_head1.shape, dtype=torch.bool), diagonal=1)
        scores_head1 = scores_head1.mask_fill(mask_head1, float("-inf"))
        # softmax
        weights_head0 = torch.softmax(scores_head0, dim=-1)
        weights_head1 = torch.softmax(scores_head1, dim=-1)
        # output
        output_head0 = weights_head0 @ V_heads[0]
        output_head1 = weights_head1 @ V_heads[1]
        output = torch.cat([output_head0, output_head1], dim=-1)
        output = self.output_project(output)
        return output
