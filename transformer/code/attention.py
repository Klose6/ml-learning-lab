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
