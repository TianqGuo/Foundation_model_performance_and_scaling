"""
Debug script to understand the BF16 forward memory issue.
"""

import torch
from cs336_systems.profiling_benchmarking.benchmark import MODEL_CONFIGS, create_model, generate_random_batch

# Clear everything first
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

config = MODEL_CONFIGS["2.7B"]
model = create_model(config, 512, device="cuda")
input_ids = generate_random_batch(4, 512, config.vocab_size, device="cuda")

print("=" * 80)
print("FP32 Forward Pass")
print("=" * 80)

model.eval()

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids)
        torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
initial_fp32 = torch.cuda.memory_allocated() / (1024 ** 2)

# Profile
with torch.no_grad():
    for _ in range(3):
        output = model(input_ids)
        torch.cuda.synchronize()

peak_fp32 = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Initial memory: {initial_fp32:.2f} MB")
print(f"Peak memory: {peak_fp32:.2f} MB")
print(f"Increase: {peak_fp32 - initial_fp32:.2f} MB")
print()

# Clean up
del model, input_ids, output
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print("Waiting 3 seconds for cleanup...")
import time
time.sleep(3)

# Create fresh model for BF16
print("=" * 80)
print("BF16 Forward Pass (Fresh)")
print("=" * 80)

model = create_model(config, 512, device="cuda")
input_ids = generate_random_batch(4, 512, config.vocab_size, device="cuda")
model.eval()

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

# Warmup
with torch.no_grad():
    for _ in range(3):
        with autocast_ctx:
            _ = model(input_ids)
        torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
initial_bf16 = torch.cuda.memory_allocated() / (1024 ** 2)

# Profile
with torch.no_grad():
    for _ in range(3):
        with autocast_ctx:
            output = model(input_ids)
        torch.cuda.synchronize()

peak_bf16 = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Initial memory: {initial_bf16:.2f} MB")
print(f"Peak memory: {peak_bf16:.2f} MB")
print(f"Increase: {peak_bf16 - initial_bf16:.2f} MB")
print()

print("=" * 80)
print("COMPARISON")
print("=" * 80)
print(f"FP32 peak: {peak_fp32:.2f} MB")
print(f"BF16 peak: {peak_bf16:.2f} MB")
print(f"Difference: {peak_bf16 - peak_fp32:.2f} MB ({(peak_bf16/peak_fp32 - 1) * 100:+.1f}%)")
print()
print("Expected: BF16 should be LOWER (activations are half the size)")
print("If BF16 is higher, there may be additional overhead from type conversions")
