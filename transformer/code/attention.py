'''
Transformer implementation with PyTorch
'''

import torch

seed_size = 42
torch.manual_seed(seed_size)

x = torch.randn(3, 4)
print(x)
print(x.shape)