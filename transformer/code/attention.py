'''
Transformer implementation with PyTorch
'''

import torch
import math

class MultiHeadAttention(torch.nn.Module):
    '''
    Multi-head attention is largely about looking at the same tokens through several different learned “lenses” at once, 
    instead of forcing one attention pattern to do everything
    '''
    def __init__(self, d_model: int = 8, num_heads: int = 2):
        # Initializes PyTorch's nn.Module base class
        super().__init__()
        # Each head gets an equal slice of the embedding
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        # Size of each head's subspace
        self.head_dim = self.d_model // self.num_heads

        self.WQ = torch.nn.Linear(d_model, d_model)
        self.WK = torch.nn.Linear(d_model, d_model)
        self.WV = torch.nn.Linear(d_model, d_model)
        self.output_project = torch.nn.Linear(d_model, d_model)

    def forward(self, X):
        batch_size, seq_len, _ = X.shape
        Q = self.WQ(X)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)
        K = self.WK(X)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)
        V = self.WV(X)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.transpose(1, 2)
        # process heads
        scores = Q @ K.transpose(-2, -1)
        # without it, dot products grow with dimension and softmax becomes very sharp (near one-hot). 
        # Scaling keeps gradients stable
        scores /= math.sqrt(self.head_dim)
        # causal mask: Upper triangle (strictly above diagonal) set to -inf -> weight 0
        # skip device setting: device=X.device for device portability
        mask = torch.triu(torch.ones(scores.shape, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        # softmax
        weights = torch.softmax(scores, dim=-1)
        # output: Swap heads and sequence back, then flatten head dimensions
        output = weights @ V
        output = output.transpose(1, 2)
        output = output.reshape(batch_size, seq_len, self.d_model)
        # Swap heads and sequence back, then flatten head dimensions
        output = self.output_project(output)
        return output

mha = MultiHeadAttention()