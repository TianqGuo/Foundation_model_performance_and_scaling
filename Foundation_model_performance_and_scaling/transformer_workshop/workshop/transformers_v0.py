"""
Simplest version of Transformers
Implement a forward pass for transformer
""" 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

B = 8 # batch size
L = 16 # max sequence length
E = 64 # embedding dimension
H = 4 # attention head count

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=2, dim_k=96, dim_v=96, dim_q=96):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = dim_q



    def forward(self, x):
        # return 1
        attention_out = self.multi_head_attention(x)
        return attention_out

    def multi_head_attention(self, inputs):
        k1, v1, q1 = self.k1(inputs), self.v1(inputs), self.q1(inputs)
        k2, v2, q2 = self.k2(inputs), self.v2(inputs), self.q2(inputs)

        head1_output = self.softmax(q1 @ k1.transpose(-2, -1) / self.dim_k ** 0.5) @ v1
        head2_output = self.softmax(q2 @ k2.transpose(-2, -1) / self.dim_k ** 0.5) @ v2

        combined_heads = torch.cat((head1_output, head2_output), dim=-1)
        attention_output = self.attention_head_projection(combined_heads)

        outputs = self.norm_mh(inputs + attention_output)

        return outputs


class MLP(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=2, dim_feedforward=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.hidden_dim)
        )
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        # pass

    def forward(self, x):
        # return 1 # place holder
        ff_out = self.ff(x)
        outputs = self.norm_ff(x + ff_out)
        return outputs


class Transformers(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = CausalSelfAttention()
        self.mlp = MLP()
    
    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x

if __name__ == "__main__":
    # B-> batch size
    # L -> seq length
    # E -> vocab embedding dimension
    # INPUT: B, L, E
    # OUTPUT: B, L, E
    batched_input = torch.randn(B, L, E)
    trans = Transformers()
    output = trans(batched_input)
    print(output.shape)