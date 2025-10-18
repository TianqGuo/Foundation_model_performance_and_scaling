================================================================================
OpenWebText Experiment - README
================================================================================

OVERVIEW
--------
This experiment trains the same Transformer architecture on OpenWebText that was
used for TinyStories, with the same compute budget (327.68M tokens).

KEY DIFFERENCES FROM TINYSTORIES
----------------------------------
1. Vocabulary: 32,000 tokens (vs 10,000 for TinyStories)
2. Model size: ~45M parameters (vs ~23M for TinyStories)
   - Larger embedding layer due to 3.2× larger vocabulary
   - Same d_model, num_layers, num_heads, d_ff
3. Data complexity: Web-crawled data (more varied, complex, realistic)
4. Dataset size: 2.7B tokens available (vs 328M for TinyStories)

CONFIGURATION
-------------
Model Architecture:
  - vocab_size: 32,000
  - context_length: 256
  - d_model: 512
  - num_layers: 4
  - num_heads: 16
  - d_ff: 1344 (SwiGLU)
  - Total parameters: 45,224,448

Training Setup:
  - Batch size: 32
  - Max iterations: 40,000
  - Learning rate: 3e-4 → 3e-5 (cosine decay)
  - Warmup: 400 iterations
  - Total tokens: 327,680,000 (same as TinyStories)

RUNNING THE EXPERIMENT
-----------------------
From project root:

uv run python cs336_basics/basics/run_experiment.py \
    --name openwebtext_main \
    --config cs336_basics/assignment_experiments/openwebtext_experiment/config_owt.json \
    --output-dir cs336_basics/basics/runs/openwebtext

EXPECTED RESULTS
----------------
- Training time: ~3-4 hours on H100 GPU
- Final validation loss: Expected to be HIGHER than TinyStories
  - TinyStories achieves ~1.39 validation loss
  - OpenWebText expected ~4-5 validation loss (harder task)
- Perplexity: Much higher than TinyStories

WHY HIGHER LOSS?
----------------
1. More complex data (web text vs children's stories)
2. Larger vocabulary (harder next-token prediction)
3. More diverse topics and writing styles
4. Same model capacity spread over harder task

OUTPUT LOCATION
---------------
cs336_basics/basics/runs/openwebtext/
├── config.json          - Experiment configuration
├── summary.json         - Final statistics
├── metrics.csv          - Training metrics
├── loss_curves.png      - Visualizations
└── checkpoints/
    ├── best_model.pt
    └── final_checkpoint.pt

NEXT STEPS
----------
1. Wait for training to complete (~3-4 hours)
2. Generate text samples from trained model
3. Compare with TinyStories results
4. Analyze why OpenWebText is more challenging

================================================================================
