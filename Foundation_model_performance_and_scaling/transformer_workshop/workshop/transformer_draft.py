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
    def __init__(self, hidden_dim=128, num_heads=2):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_mh = nn.LayerNorm(hidden_dim)

        self.register_buffer("causal_mask", None)

        # pass

    def _get_causal_mask(self, seq_len, device):
        """Generate or retrieve causal mask"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create lower triangular mask (1s below diagonal, 0s above)
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            self.register_buffer("causal_mask", mask)
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x):
        attention_out = self.multi_head_attention(x)
        return attention_out

    def multi_head_attention(self, inputs):
        batch_size, seq_len, hidden_dim = inputs.shape

        # Generate Q, K, V for all heads at once
        Q = self.q_proj(inputs)  # (B, L, hidden_dim)
        K = self.k_proj(inputs)  # (B, L, hidden_dim)
        V = self.v_proj(inputs)  # (B, L, hidden_dim)

        # Reshape for multi-head attention: (B, L, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        causal_mask = self._get_causal_mask(seq_len, inputs.device)
        # Convert mask to attention mask (0 -> -inf, 1 -> 0)
        attention_mask = (1 - causal_mask) * -1e9
        attention_scores = attention_scores + attention_mask.unsqueeze(0).unsqueeze(0)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, L, head_dim)

        # Transpose back and reshape: (B, L, hidden_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )

        # Apply output projection
        attention_output = self.out_proj(attention_output)

        # Add residual connection and layer norm
        outputs = self.norm_mh(inputs + attention_output)

        return outputs


class MLP(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=2, dim_feedforward=2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, self.hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        
        # pass

    def forward(self, x):
        ff_outputs = self.ff(x)
        ff_norm = self.norm_ff(ff_outputs)
        return ff_norm
        # return 1 # place holder


class Transformers(nn.Module):
    def __init__(self):
        super.__init__()
        self.attention = CausalSelfAttention()
        self.mlp = MLP()
        # pass

    def forward(self, x):
        x = self.attention(x)
        x = self.mlp(x)
        # pass



if __name__ == "__main__":
    # B-> batch size
    # L -> seq length
    # E -> vocab embedding dimension
    # INPUT: B, L, E
    # OUTPUT: B, L, E
    cur_inputs = torch.randn(B, L, E)
    cur_transformer = CausalSelfAttention()
    cur_outputs = cur_transformer(cur_inputs)

    print(f"Input shape: {cur_inputs.shape}")
    print(f"Output shape: {cur_outputs.shape}")
    print("✓ Success!")


