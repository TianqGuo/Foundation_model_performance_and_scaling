"""
Simplest version of Transformers
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
    """
    Input -> q, k, v projection -> attention -> cross-head projection
    """
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(E, 3 * E, bias=False)
        self.out_proj = nn.Linear(E, E, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(1, 1, L, L, dtype=bool)),  # max seq len
            persistent=False  # not be saved in state_dict in checkpoint
        )

    def forward(self, x):
        B, T, E = x.size()
        assert E % H == 0
        x = self.qkv_proj(x)  # [B, T, E] -> [B, T, 3E]
        q, k, v = x.split(E, dim=-1)  # [B, T, 3E] -> 3 of [B, T, E]
        q = q.view(B, T, H, E // H).transpose(1, 2)  # [B, T, E] -> [B, H, T, E/H]
        k = k.view(B, T, H, E // H).transpose(1, 2)  # [B, T, E] -> [B, H, T, E/H]
        v = v.view(B, T, H, E // H).transpose(1, 2)  # [B, T, E] -> [B, H, T, E/H]

        scores = q @ k.transpose(-2, -1) / math.sqrt(E // H)  # [B, H, T, E/H] @ [B, H, E/H, T] -> [B, H, T, T]

        scores = scores.masked_fill(~self.mask[:, :, :T, :T], float('-inf'))  # apply causal mask

        attn = F.softmax(scores, dim=-1)  # [B, H, T, T]
        attn = attn @ v  # [B, H, T, T] @ [B, H, T, E/H] -> [B, H, T, E/H]
        attn = attn.transpose(1, 2).contiguous().view(B, T, E)  # [B, H, T, E/H] -> [B, T, E]
        x = self.out_proj(attn)  # [B, T, E] -> [B, T, E]
        return x


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # [B, T, E] ->
        self.fc = nn.Sequential(
            nn.Linear(E, 4 * E, bias=False),
            nn.GELU(),
            nn.Linear(4 * E, E, bias=False),
        )


    def forward(self, x):
        return self.fc(x)

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
    batched_input = torch.randn(B, L, E)
    trans = Transformers()
    output = trans(batched_input)
    print(output.shape)