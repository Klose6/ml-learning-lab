'''
Transformer implementation with PyTorch
'''

import torch

seed_size = 42
torch.manual_seed(seed_size)

# Implement the attention formula
X = torch.randn(3, 4)
Q = X
K = X
V = X
# Calculate the attention scores
scores = Q@K.T
# Scale
d_k = K.size(-1)
scaled_scores = scores / (d_k ** 0.5)
# Run softmax on the last dimension(-1) of the tensor
weights = torch.softmax(scaled_scores, dim=-1)
# Weighted sum of values
output = weights@V

print(output)
print(output.shape)