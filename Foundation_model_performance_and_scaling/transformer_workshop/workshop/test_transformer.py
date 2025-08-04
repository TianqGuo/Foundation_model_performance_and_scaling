"""
Test script to verify the transformer implementation
"""
import torch
import torch.nn.functional as F
from transformers_v0 import Transformers, CausalSelfAttention

def test_flexible_heads():
    """Test that the transformer works with different numbers of heads"""
    print("Testing flexible attention heads...")
    
    batch_size, seq_len, hidden_dim = 2, 8, 64
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Test different head counts
    head_counts = [1, 2, 4, 8]
    
    for num_heads in head_counts:
        print(f"  Testing {num_heads} heads...")
        transformer = Transformers(hidden_dim=hidden_dim, num_heads=num_heads)
        
        try:
            output = transformer(input_tensor)
            assert output.shape == input_tensor.shape, f"Output shape mismatch: {output.shape} vs {input_tensor.shape}"
            print(f"    ✓ {num_heads} heads: Output shape {output.shape}")
        except Exception as e:
            print(f"    ✗ {num_heads} heads failed: {e}")
            return False
    
    return True

def test_causal_masking():
    """Test that causal masking is working correctly"""
    print("\nTesting causal masking...")
    
    batch_size, seq_len, hidden_dim = 1, 4, 8
    num_heads = 2
    
    # Create attention layer
    attention = CausalSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # Create simple input where we can track attention patterns
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Get the causal mask
    device = input_tensor.device
    causal_mask = attention._get_causal_mask(seq_len, device)
    
    print(f"  Causal mask shape: {causal_mask.shape}")
    print(f"  Causal mask:\n{causal_mask}")
    
    # Verify mask is lower triangular
    expected_mask = torch.tril(torch.ones(seq_len, seq_len))
    if torch.equal(causal_mask, expected_mask):
        print("    ✓ Causal mask is correct (lower triangular)")
    else:
        print("    ✗ Causal mask is incorrect")
        return False
    
    # Test forward pass
    try:
        output = attention(input_tensor)
        assert output.shape == input_tensor.shape, f"Output shape mismatch"
        print(f"    ✓ Forward pass successful: {output.shape}")
    except Exception as e:
        print(f"    ✗ Forward pass failed: {e}")
        return False
    
    return True

def test_attention_scores_masking():
    """Test that attention scores are properly masked"""
    print("\nTesting attention score masking...")
    
    batch_size, seq_len, hidden_dim = 1, 3, 4
    num_heads = 1
    
    attention = CausalSelfAttention(hidden_dim=hidden_dim, num_heads=num_heads)
    
    # Create a simple input
    input_tensor = torch.ones(batch_size, seq_len, hidden_dim)
    
    # Manually compute what should happen
    with torch.no_grad():
        Q = attention.q_proj(input_tensor)
        K = attention.k_proj(input_tensor)
        
        Q = Q.view(batch_size, seq_len, num_heads, hidden_dim // num_heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, num_heads, hidden_dim // num_heads).transpose(1, 2)
        
        # Compute raw attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_dim // num_heads) ** 0.5
        
        # Apply causal mask
        causal_mask = attention._get_causal_mask(seq_len, input_tensor.device)
        attention_mask = (1 - causal_mask) * -1e9
        masked_scores = attention_scores + attention_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply softmax
        attention_weights = F.softmax(masked_scores, dim=-1)
        
        print(f"  Raw attention scores shape: {attention_scores.shape}")
        print(f"  Attention weights after masking:\n{attention_weights[0, 0]}")
        
        # Check that upper triangular part has very small weights (due to masking)
        weights_2d = attention_weights[0, 0]  # First batch, first head
        
        # Check that position 0 can only attend to itself
        if weights_2d[0, 1] < 1e-6 and weights_2d[0, 2] < 1e-6:
            print("    ✓ Position 0 correctly masked from future positions")
        else:
            print("    ✗ Position 0 can attend to future positions")
            return False
            
        # Check that position 1 can attend to 0 and 1, but not 2
        if weights_2d[1, 2] < 1e-6 and weights_2d[1, 0] > 1e-6:
            print("    ✓ Position 1 correctly masked from future positions")
        else:
            print("    ✗ Position 1 masking incorrect")
            return False
    
    return True

def main():
    print("=== Transformer Implementation Verification ===\n")
    
    tests = [
        test_flexible_heads,
        test_causal_masking,
        test_attention_scores_masking
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\n=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("🎉 All tests passed! The implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()