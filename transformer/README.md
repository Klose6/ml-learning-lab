# Transformer

A from-scratch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), built with PyTorch as part of [ml-learning-lab](../README.md).

## Overview

This module walks through the core components of the original Transformer, starting with scaled dot-product attention and building up to a full encoder-decoder model.

## Scaled Dot-Product Attention

The attention mechanism maps a query and a set of key-value pairs to an output:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:

- **Q** — query matrix
- **K** — key matrix
- **V** — value matrix
- **d_k** — dimension of the key vectors (used for scaling)

## Implementation Checklist

- [ ] Self-attention
- [ ] Multi-head attention
- [ ] Position-wise feed-forward network
- [ ] Positional encoding
- [ ] Encoder block
- [ ] Decoder block
- [ ] Full Transformer

## Project Structure

```
transformer/
├── README.md
└── code/
    └── attention.py    # Scaled dot-product attention
```

## Getting Started

**Requirements:** Python >= 3.13, PyTorch (see [pyproject.toml](../pyproject.toml))

```bash
# From the repository root
python transformer/code/attention.py
```

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *NeurIPS*.
