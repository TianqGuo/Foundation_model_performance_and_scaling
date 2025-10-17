# Text Generation with Trained Transformer Models

This directory contains tools and results for generating text from trained Transformer language models.

## Files

- **`generate_text.py`** - Main script for text generation
- **`TEXT_GENERATION_RESULTS.txt`** - Comprehensive results and analysis
- **`generation_results.txt`** - Additional sample generations

## Quick Start

### Basic Usage

Generate text from the best trained model (batch_8):

```bash
uv run python cs336_basics/assignment_experiments/generate_text.py
```

### Custom Prompts

```bash
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --prompts "Once upon a time" "In a magical forest"
```

### Adjust Generation Parameters

```bash
# Lower temperature for more focused/coherent text
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --temperature 0.7

# Higher temperature for more creative text
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --temperature 1.2

# Use top-p (nucleus) sampling
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --top-p 0.9

# Use top-k sampling
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --top-k 50

# Generate longer sequences
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --max-tokens 500
```

### Use Different Checkpoint

```bash
# Use learning rate sweep checkpoint
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --checkpoint cs336_basics/basics/runs/lr_sweep/lr_3e_04/checkpoints/best_model.pt

# Use different batch size model
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --checkpoint cs336_basics/basics/runs/batch_size_sweep/batch_32/checkpoints/best_model.pt
```

### Save Output to File

```bash
uv run python cs336_basics/assignment_experiments/generate_text.py \
    --output my_generations.txt \
    --prompts "Story prompt 1" "Story prompt 2"
```

## Generation Parameters

### Temperature
- **Range:** 0.1 - 2.0 (typical)
- **Lower (0.5-0.8):** More deterministic, coherent, repetitive
- **Medium (0.8-1.0):** Balanced creativity and coherence ← **RECOMMENDED**
- **Higher (1.0-1.5):** More creative, diverse, potentially less coherent

### Top-k Sampling
- **Range:** 10 - 100 (typical)
- Limits sampling to the k most likely tokens
- Lower values → more focused, higher values → more diverse

### Top-p (Nucleus) Sampling
- **Range:** 0.5 - 0.99
- Samples from smallest set of tokens whose cumulative probability ≥ p
- 0.9-0.95 recommended for good balance

### Max Tokens
- Maximum number of new tokens to generate
- Default: 256
- Model context length: 256 tokens total (prompt + generated)

## Best Practices

1. **Start with defaults:** `temperature=0.9, max_tokens=256`
2. **For coherent stories:** Use `temperature=0.8` without top-k/top-p
3. **For creative variety:** Use `temperature=1.0 --top-p 0.95`
4. **For consistent results:** Set `--seed 42`

## Model Information

Default model: `batch_8/checkpoints/best_model.pt`
- **Architecture:** 4-layer Transformer, 16 attention heads
- **Parameters:** 22,696,448
- **Vocabulary:** 10,000 tokens (BPE)
- **Context Length:** 256 tokens
- **Validation Loss:** 1.3199 (best among all experiments)
- **Training Data:** TinyStories (~328M tokens processed)

## Example Output

**Prompt:** "Once upon a time, there"

**Generated (temperature=0.8):**
> Once upon a time, there was a little girl named Sue. She had a big red box that she found in her room. Sue was very excited to open it up.
> Inside the box, Sue found a pretty doll. The doll had a harsh heart and a sad face. Sue decided to show the doll to her mom. Her mom said, "Sue, this doll is not nice. It needs a friend." So, Sue took the doll to her mom and showed it to her mom.
> Her mom said, "Let's take the doll to the park." They took the doll to the park and played with it. They had a lot of fun. Sue and the doll became best friends. From that day on, Sue always took good care of her doll.
> <|endoftext|>

## Troubleshooting

### CUDA Out of Memory
- Generation uses minimal memory, but if issues occur:
  - Use `--device cpu` to run on CPU
  - Reduce `--max-tokens`

### Import Errors
- Ensure you're running from the project root
- Use `uv run python` instead of `python` directly

### Poor Quality Output
- Check that you're using a well-trained checkpoint
- Verify validation loss < 1.5 in the checkpoint's summary.json
- Adjust temperature (try 0.8-0.9)

## Related Experiments

- **Learning Rate Sweep:** `cs336_basics/assignment_experiments/lr_sweep/`
- **Batch Size Sweep:** `cs336_basics/assignment_experiments/batch_size_experiment/`

See the respective directories for analysis and results from hyperparameter tuning experiments.
