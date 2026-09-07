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

        outputs = []
        # Processing the multiple heads
        for q, k, v in zip(Q_heads, K_heads, V_heads):
            # process heads
            scores = q @ k.T
            scores /= math.sqrt(self.head_dim)
            # causal mask
            mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
            scores = scores.mask_fill(mask, float("-inf"))
            # softmax
            weights= torch.softmax(scores, dim=-1)
            # output
            output_head = weights @ v
            outputs.append(output_head)

        output = torch.cat(outputs, dim=-1)
        output = self.output_project(output)
        return output
