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
scores = Q@K.T

d_k = K.size(-1)

scaled_scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

print(scaled_scores)
print(scaled_scores.shape)