# Verification Commands

Run these commands in your PyTorch environment to verify the implementation:

## 1. Basic functionality test:
```bash
cd transformer_workshop/workshop
python transformers_v0.py
```

## 2. Comprehensive test suite:
```bash
python test_transformer.py
```

## 3. Quick manual verification in Python:
```python
import torch
from transformers_v0 import Transformers, CausalSelfAttention

# Test 1: Basic forward pass
input_tensor = torch.randn(2, 8, 64)  # batch=2, seq_len=8, hidden_dim=64
transformer = Transformers(hidden_dim=64, num_heads=4)
output = transformer(input_tensor)
print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print("✓ Basic forward pass works" if output.shape == input_tensor.shape else "✗ Shape mismatch")

# Test 2: Different head counts
for heads in [1, 2, 4, 8]:
    try:
        t = Transformers(hidden_dim=64, num_heads=heads)
        out = t(input_tensor)
        print(f"✓ {heads} heads: {out.shape}")
    except Exception as e:
        print(f"✗ {heads} heads failed: {e}")

# Test 3: Causal mask verification
attention = CausalSelfAttention(hidden_dim=64, num_heads=4)
mask = attention._get_causal_mask(4, torch.device('cpu'))
print(f"Causal mask:\n{mask}")
print("✓ Causal mask is lower triangular" if torch.equal(mask, torch.tril(torch.ones(4, 4))) else "✗ Mask incorrect")
```

## Expected Results:

1. **transformers_v0.py** should output: `torch.Size([8, 16, 64])`

2. **test_transformer.py** should show:
   - ✓ All head counts (1, 2, 4, 8) working
   - ✓ Causal mask is lower triangular
   - ✓ Attention properly masked from future positions

3. **Manual verification** should show:
   - Input/output shapes match
   - All head counts work without errors
   - Causal mask looks like:
     ```
     tensor([[1., 0., 0., 0.],
             [1., 1., 0., 0.],
             [1., 1., 1., 0.],
             [1., 1., 1., 1.]])
     ```

If any test fails, let me know the error message and I'll help fix it!