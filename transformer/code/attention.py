'''
Transformer implementation with PyTorch
'''

import torch
import math

seed_size = 42
torch.manual_seed(seed_size)

# Implement the attention formula
X = torch.randn(3, 4)
# Learnable projection
WQ = torch.nn.Linear(4, 4)
WK = torch.nn.Linear(4, 4)
WV = torch.nn.Linear(4, 4)

Q = WQ(X)
K = WK(X)
V = WV(X)
# Calculate the attention scores
scores = Q@K.T
# Scale
d_k = K.size(-1)
scaled_scores = scores / math.sqrt(d_k)
# Mask the top half for causal attention
r, c = torch.triu_indices(scaled_scores.size(0), scaled_scores.size(1), offset=1)
scaled_scores[r, c] = float("-inf")
# Run softmax on the last dimension(-1) of the tensor
weights = torch.softmax(scaled_scores, dim=-1)
# Weighted sum of values
output = weights@V

print(weights)
print(output)
