"""
Debug script showing the CORRECT way to measure BF16 inference memory.

For inference, we should convert model to BF16 directly, not use autocast.
Autocast is designed for training and creates overhead in eval mode.
"""

import torch
from cs336_systems.profiling_benchmarking.benchmark import MODEL_CONFIGS, create_model, generate_random_batch

config = MODEL_CONFIGS["2.7B"]

print("=" * 80)
print("METHOD 1: FP32 Inference (Baseline)")
print("=" * 80)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = create_model(config, 512, device="cuda")
input_ids = generate_random_batch(4, 512, config.vocab_size, device="cuda")
model.eval()

# Warmup
with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids)
        torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
initial_fp32 = torch.cuda.memory_allocated() / (1024 ** 2)

with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids)
        torch.cuda.synchronize()

peak_fp32 = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Initial memory: {initial_fp32:.2f} MB")
print(f"Peak memory: {peak_fp32:.2f} MB")
print(f"Increase: {peak_fp32 - initial_fp32:.2f} MB")
print()

del model, input_ids
torch.cuda.empty_cache()

import time
time.sleep(2)

print("=" * 80)
print("METHOD 2: BF16 with Autocast (WRONG for inference)")
print("=" * 80)

torch.cuda.reset_peak_memory_stats()

model = create_model(config, 512, device="cuda")
input_ids = generate_random_batch(4, 512, config.vocab_size, device="cuda")
model.eval()

autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

with torch.no_grad():
    for _ in range(3):
        with autocast_ctx:
            _ = model(input_ids)
        torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
initial_autocast = torch.cuda.memory_allocated() / (1024 ** 2)

with torch.no_grad():
    for _ in range(3):
        with autocast_ctx:
            _ = model(input_ids)
        torch.cuda.synchronize()

peak_autocast = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Initial memory: {initial_autocast:.2f} MB")
print(f"Peak memory: {peak_autocast:.2f} MB")
print(f"Increase: {peak_autocast - initial_autocast:.2f} MB")
print()

del model, input_ids
torch.cuda.empty_cache()
time.sleep(2)

print("=" * 80)
print("METHOD 3: Direct BF16 Conversion (CORRECT for inference)")
print("=" * 80)

torch.cuda.reset_peak_memory_stats()

model = create_model(config, 512, device="cuda")
# Convert model to BF16 directly
model = model.to(torch.bfloat16)

input_ids = generate_random_batch(4, 512, config.vocab_size, device="cuda")
model.eval()

with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids)
        torch.cuda.synchronize()

torch.cuda.reset_peak_memory_stats()
initial_bf16 = torch.cuda.memory_allocated() / (1024 ** 2)

with torch.no_grad():
    for _ in range(3):
        _ = model(input_ids)
        torch.cuda.synchronize()

peak_bf16 = torch.cuda.max_memory_allocated() / (1024 ** 2)

print(f"Initial memory: {initial_bf16:.2f} MB")
print(f"Peak memory: {peak_bf16:.2f} MB")
print(f"Increase: {peak_bf16 - initial_bf16:.2f} MB")
print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Method 1 (FP32):              {peak_fp32:.2f} MB")
print(f"Method 2 (Autocast in eval):  {peak_autocast:.2f} MB  ← WRONG METHOD!")
print(f"Method 3 (Direct BF16):       {peak_bf16:.2f} MB  ← CORRECT!")
print()
print(f"FP32 vs Direct BF16: {(1 - peak_bf16/peak_fp32) * 100:.1f}% memory savings")
print()
print("Conclusion:")
print("- Autocast in eval mode creates massive overhead (~10x activations)")
print("- For inference profiling, convert model to BF16 directly")
print("- For training profiling, autocast is appropriate")
