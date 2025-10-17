# Batch Size Experiment Guide

This guide distills the Assignment 1 instructions relevant to the
`batch_size_experiment` sweep and adds practical tips gathered while tuning
the TinyStories transformer.

## Assignment Requirements

- **Goal:** Vary batch size from 1 up to your GPU limit. Include intermediate
  values such as 64 and 128.
- **Deliverables:**
  - Learning curves for several batch sizes.
  - Brief discussion summarizing how batch size affects throughput,
    convergence speed, and final validation loss. Re-tune the learning rate if
    necessary.
- **Reference Model Hyperparameters:**
  - `vocab_size = 10_000`
  - `context_length = 256`
  - `d_model = 512`
  - `d_ff = 1344`
  - `num_layers = 4`, `num_heads = 16`
  - `rope_theta = 10_000`
  - Total tokens processed ≈ `327_680_000`

## Recommended Workflow

1. **Start from the learning-rate sweep defaults.** Reuse the same optimizer
   settings unless you see instability.
2. **Pick a baseline batch size** (e.g., 128) and reproduce a reference run to
   confirm throughput remains high (~90k tok/s on 40-series GPUs with
   TF32 + `torch.compile`).
3. **Scale batch sizes** geometrically (1, 8, 32, 64, 128, 256, 512…) until you
   hit memory limits. The sweep script keeps total tokens constant by reducing
   the number of steps.
4. **Track metrics:**
   - Throughput (`Tok/s`) – expect higher values for larger batches.
   - Validation loss trajectories – look for plateaus or degraded quality.
   - Training stability – if runs diverge, lower the learning rate or warmup.
5. **Document findings** in `batch_size_experiment/notes.md` or your report.

## Troubleshooting

- **Out of Memory:** Reduce batch size or enable gradient accumulation
  (requires code changes in `run_experiment.py`). The sweep script will catch
  CUDA OOM errors and mark the run as failed.
- **Divergence at large batches:** Consider lowering the learning rate
  proportionally (linear scaling rule) or extending warmup.
- **Low throughput:** Verify TF32 and compilation are enabled (see
  `lr_sweep/test_performance.py`). Always re-run the performance test if you
  modify kernels or reinstall PyTorch.

## Low-Resource Settings

When running on CPU or Apple Silicon (MPS):

- Reduce total tokens to ~40M (handled automatically with `--low-resource`).
- Target validation loss ≤ 2.0.
- Skip TF32 on MPS – stick with the eager backend.

## Next Steps

- Add your analysis script (e.g., `analyze_batch_sizes.py`) to aggregate metrics
  and produce the deliverable plots.
- Integrate with any external experiment tracking tools (Weights & Biases,
  TensorBoard) if desired.

Happy experimenting!
