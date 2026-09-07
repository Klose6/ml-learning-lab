'''
Transformer implementation with PyTorch
'''

import torch

seed_size = 42
torch.manual_seed(seed_size)

# implement the attention formula
X = torch.randn(3, 4)
Q = X
K = X
V = X
scores = Q@K.T

d_k = K.size(-1)

scaled_scores = scores / d_k ** 0.5

weights = torch.softmax(scaled_scores, dim=-1)
output = weights@V
print(output)
print(output.shape)