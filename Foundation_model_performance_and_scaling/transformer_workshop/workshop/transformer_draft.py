"""
Simplest version of Transformers
Implement a forward pass for transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

B = 8  # batch size
L = 16  # max sequence length
E = 64  # embedding dimension
H = 4  # attention head count


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def _get_causal_mask(self, seq_len, device):
        """Generate or retrieve causal mask"""
        pass

    def forward(self, x):
        pass

    def multi_head_attention(self, inputs):
        pass


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        pass

    def forward(self, x):
        return 1 # place holder


class Transformers(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


if __name__ == "__main__":
    # B-> batch size
    # L -> seq length
    # E -> vocab embedding dimension
    # INPUT: B, L, E
    # OUTPUT: B, L, E
