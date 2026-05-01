# Transformer Ablation Study - Detailed Results

## Executive Summary

This report presents comprehensive results from ablation experiments on a 22.7M parameter Transformer language model trained on TinyStories. We investigate the impact of four key architectural components: RMSNorm, pre-norm vs post-norm placement, RoPE position embeddings, and SwiGLU vs SiLU feed-forward networks.

**Key Findings:**
- All components have relatively small individual impact (<2% validation loss change)
- SwiGLU provides the largest benefit (+1.43% validation loss when removed)
- RMSNorm removal causes training instability (0.60% degradation despite convergence)
- RoPE position embeddings have negligible impact at this scale (0.00% difference)
- Post-norm architecture performs slightly better than pre-norm (-0.03% improvement)

---

## Experiment Configuration

### Model Architecture
- **Parameters:** 22.7M trainable parameters
- **Dimensions:** d_model=512, d_ff=1344 (SwiGLU) or 2048 (SiLU)
- **Layers:** 4 Transformer blocks
- **Heads:** 16 attention heads per layer
- **Context Length:** 256 tokens
- **Vocabulary:** 10,000 tokens (TinyStories)

### Training Setup
- **Dataset:** TinyStories (~328M tokens processed)
- **Batch Size:** 32
- **Training Steps:** 40,000 iterations
- **Learning Rate:** 3e-4 (cosine decay to 3e-5)
- **Warmup:** 400 iterations
- **Weight Decay:** 0.1
- **Gradient Clipping:** 1.0
- **Optimizer:** AdamW (β1=0.9, β2=0.999)

---

## Results Overview

| Ablation | Best Val Loss | Δ vs Baseline | Train Time (h) | Component Tested |
|----------|--------------|---------------|----------------|------------------|
| **Baseline** | **1.3902** | **+0.00%** | 2.10 | Full model (pre-norm, RMSNorm, SwiGLU, RoPE) |
| Post-norm | 1.3897 | -0.03% | 2.21 | Norm placement |
| NoPE | 1.3902 | +0.00% | 2.20 | Position encoding |
| No Layer Norm | 1.3985 | +0.60% | 2.02 | RMSNorm importance |
| SiLU FFN | 1.4101 | +1.43% | 2.05 | Feed-forward gating |

---

## Detailed Ablation Analysis

### 1. Layer Norm Ablation (Problem: layer_norm_ablation)

**Configuration:** Removed all RMSNorm layers from the model

**Results:**
- **Best Validation Loss:** 1.3985
- **Degradation vs Baseline:** +0.60% (+0.0083 absolute)
- **Training Stability:** Model trained successfully at LR=3e-4 but showed higher initial loss (17.68 vs 9.26)

**Analysis:**

The removal of RMSNorm layers had a measurable but relatively modest impact on final performance. However, the training dynamics revealed important insights:

1. **Training Instability:** Initial loss was nearly 2× higher (17.68 vs 9.26), indicating significantly degraded gradient flow early in training
2. **Successful Convergence:** Despite the unstable start, the model converged successfully without requiring learning rate reduction
3. **Performance Degradation:** Final validation loss increased by 0.60%, showing that while not critical for convergence, RMSNorm provides measurable benefits

**Key Observation:** At the default learning rate (3e-4), the no-layer-norm model trained successfully, contrary to the common belief that layer normalization is strictly necessary for stable training. However, the degraded performance and unstable early training demonstrate RMSNorm's value.

**Commentary:**
RMSNorm is important but not strictly necessary for training modern Transformers at this scale. Its primary benefits are:
1. Stabilizing gradient flow (evidenced by lower initial loss)
2. Enabling faster/more stable convergence
3. Providing small but consistent performance improvements (~0.6%)

For larger models or higher learning rates, RMSNorm likely becomes more critical.

---

### 2. Pre-norm vs Post-norm Ablation (Problem: pre_norm_ablation)

**Configuration:** Moved normalization from before (pre-norm) to after (post-norm) attention and FFN blocks

**Results:**
- **Best Validation Loss:** 1.3897
- **Difference vs Baseline:** -0.03% (-0.0005 absolute)
- **Training Time:** 2.21 hours (5% slower than baseline)

**Analysis:**

The post-norm architecture performed virtually identically to pre-norm, with a negligible 0.03% improvement that falls within expected variance.

**Comparative Learning Curves:**
- Both architectures showed identical initial loss (9.26)
- Training dynamics were nearly indistinguishable
- Post-norm took slightly longer per iteration (5% increase) likely due to different computational patterns

**Historical Context:**
Pre-norm became the dominant architecture choice in modern Transformers (GPT, LLaMA) primarily because:
1. **Gradient Flow:** Pre-norm creates a direct path for gradients, improving training stability
2. **Scalability:** The benefits of pre-norm increase with model depth
3. **Initialization:** Pre-norm is less sensitive to initialization schemes

**Why Post-norm Performed Well Here:**
At 4 layers depth, gradient flow issues are minimal. Post-norm's theoretical advantage (normalizing the final output of each block) provides similar benefits to pre-norm's gradient flow at this shallow depth.

**Commentary:**
For shallow models (≤8 layers), pre-norm vs post-norm makes little practical difference. Pre-norm remains the preferred choice due to better scaling properties for deeper models. The community standard of pre-norm is well-justified for production systems but not critical at small scales.

---

### 3. No Position Embeddings Ablation (Problem: no_pos_emb)

**Configuration:** Removed RoPE (Rotary Position Embeddings), attention operates without explicit position information

**Results:**
- **Best Validation Loss:** 1.3902
- **Difference vs Baseline:** +0.00% (identical to 4 decimal places)
- **Training Dynamics:** Virtually identical to baseline

**Analysis:**

This is the most surprising result: removing position embeddings entirely had **zero measurable impact** on validation loss. The model achieved 1.3902, identical to the baseline.

**Why Position Embeddings Didn't Matter:**

1. **Implicit Position Information:** Transformers can learn positional patterns through:
   - Causal masking structure (tokens only attend to previous positions)
   - Dataset statistics (certain words/patterns appear at specific relative positions)
   - Attention pattern biases learned during training

2. **Short Context Length:** At 256 tokens, position ambiguity is less problematic than at longer contexts (e.g., 2048+ tokens)

3. **Task Characteristics:** TinyStories is a simple dataset where local dependencies matter more than long-range positional relationships

**When RoPE Matters:**
Position embeddings become critical for:
- **Long context tasks** (>1024 tokens) where relative positions are ambiguous
- **Tasks requiring precise position reasoning** (e.g., "what is the 5th word?")
- **Length generalization** (training on 256 tokens, inference at 512+)

**Commentary:**
While RoPE showed no benefit in this specific setting, it's still recommended for production systems because:
1. Provides explicit inductive bias for position
2. Enables length extrapolation
3. Zero cost (no learnable parameters)
4. Proven benefits at scale

The lack of degradation without RoPE suggests that for simple, short-context tasks, position embeddings may be unnecessary overhead.

---

### 4. SwiGLU vs SiLU Ablation (Problem: swiglu_ablation)

**Configuration:** Replaced SwiGLU (gated activation) with simple SiLU activation
- **Baseline (SwiGLU):** Two linear projections with gating: `W2(SiLU(W1(x)) ⊙ W3(x))`
- **Ablation (SiLU):** Single linear projection: `W2(SiLU(W1(x)))`
- **Parameter Matching:** Adjusted d_ff to maintain similar parameter count (1344→2048)

**Results:**
- **Best Validation Loss:** 1.4101
- **Degradation vs Baseline:** +1.43% (+0.0199 absolute)
- **Training:** Stable, no convergence issues

**Analysis:**

SwiGLU provided the **largest measurable benefit** of all ablations. The 1.43% degradation when switching to SiLU demonstrates clear value from the gating mechanism.

**Why SwiGLU Performs Better:**

1. **Gating Mechanism:** The element-wise multiplication (⊙) allows the network to dynamically control information flow:
   - `W1(x)` learns what information to process
   - `W3(x)` learns gating signals (what to let through)
   - This provides finer-grained control than simple activation functions

2. **Effective Capacity:** While parameter counts are matched, SwiGLU's gating provides higher effective capacity through more flexible computation

3. **Empirical Track Record:** SwiGLU is used in LLaMA, PaLM, and other state-of-the-art models for good reason

**Performance Impact:**
- Training loss degraded from 1.418 → 1.448 (+2.1%)
- Validation loss degraded from 1.390 → 1.410 (+1.4%)
- Generalization gap remained similar (model didn't overfit more)

**Computational Trade-offs:**
- **SwiGLU:** Higher computation (2 matrix multiplications in forward pass)
- **SiLU:** Lower computation (1 matrix multiplication)
- In this experiment, SwiGLU was only marginally slower despite 1.5× more FFN operations

**Commentary:**
SwiGLU's gating mechanism provides measurable and consistent improvements over simple activations. The 1.43% degradation is substantial at scale:
- For a model targeting 10.0 perplexity, this translates to 0.14 points worse
- When training for weeks on massive datasets, this difference justifies the minor computational overhead

The gating mechanism appears to be genuinely valuable architectural innovation, not just a hyperparameter tweak.

---

## Component Importance Ranking

Based on validation loss degradation when removed:

1. **SwiGLU** (1.43% degradation) - **MODERATE IMPORTANCE**
   - Clear performance benefit from gating mechanism
   - Recommended for production systems

2. **RMSNorm** (0.60% degradation) - **LOW-MODERATE IMPORTANCE**
   - Stabilizes training and improves convergence
   - More critical at larger scales

3. **Pre-norm vs Post-norm** (0.03% improvement) - **NEGLIGIBLE IMPORTANCE**
   - Virtually no difference at 4 layers
   - Pre-norm preferred for deeper models

4. **RoPE** (0.00% difference) - **NEGLIGIBLE IMPORTANCE**
   - No measurable benefit at this scale/task
   - Still recommended for long-context scenarios

---

## Training Dynamics Comparison

### Convergence Speed
All ablations converged at similar rates:
- Validation loss reached ~1.5 by iteration 4,000
- Final performance achieved by iteration 30,000-35,000
- No significant differences in convergence dynamics

### Training Stability
- **Most Stable:** Baseline, post-norm, no_position_emb (initial loss ~9.26)
- **Least Stable:** No layer norm (initial loss 17.68, but still converged)
- **All models:** No divergence, NaN, or training failures

### Computational Efficiency
Training time per experiment (40,000 iterations):
- **Fastest:** No layer norm (2.02h) - fewer operations
- **Slowest:** Post-norm (2.21h) - different computational pattern
- **Baseline:** 2.10h (reference)

The performance differences are minimal (<10%), suggesting computational cost is not a major factor in choosing between these architectures.

---

## Recommendations

### For Production Systems
1. **Use SwiGLU:** Clear 1.4% performance benefit with minimal cost
2. **Use RMSNorm:** Provides stability and 0.6% improvement
3. **Use Pre-norm:** Standard choice, better for deeper models
4. **Use RoPE:** No cost, enables length extrapolation, proven at scale

### For Research/Experimentation
1. **Shallow models (<8 layers):** Pre-norm vs post-norm doesn't matter
2. **Short contexts (<512 tokens):** Position embeddings may be optional
3. **Limited compute:** SiLU is acceptable compromise (saves ~1% compute)
4. **Training stability issues:** RMSNorm should be your first addition

### For This Specific Task (TinyStories, 256 context)
The minimal component configuration that maintains performance:
- **Essential:** SwiGLU (1.4% loss without it)
- **Recommended:** RMSNorm (0.6% loss without it)
- **Optional:** Pre-norm, RoPE (negligible impact)

---

## Limitations and Future Work

### Limitations of This Study
1. **Small Scale:** 22.7M parameters is tiny by modern standards (GPT-3 is 175B)
2. **Shallow Model:** 4 layers may not reveal benefits that emerge at 24+ layers
3. **Simple Dataset:** TinyStories is easier than realistic language modeling
4. **Short Context:** 256 tokens doesn't test long-range dependencies
5. **Single Seed:** Results may have some variance (though likely <0.2%)

### Future Experiments
1. **Learning Rate Sweep for No-LayerNorm:** Test if lower LR improves stability
2. **Deeper Models:** Repeat at 12-24 layers to see if patterns change
3. **Longer Context:** Test at 1024-2048 tokens for position encoding benefits
4. **Multiple Seeds:** Quantify variance to confirm statistical significance
5. **Combination Ablations:** Test removing multiple components simultaneously

---

## Conclusion

This ablation study reveals that modern Transformer architectures are remarkably robust. Removing any single component causes less than 2% performance degradation, suggesting that:

1. **Architectural synergy:** Components work together, no single component is critical
2. **Over-engineering:** At small scales, many "standard" components provide minimal benefit
3. **Scale-dependent benefits:** Some components (RMSNorm, pre-norm) likely matter more at larger scales

**Practical Takeaway:** For researchers building small-scale models, SwiGLU is the only component with substantial individual impact. However, production systems should use all standard components because:
- Combined benefits likely exceed individual measurements
- Scale changes importance (what doesn't matter at 20M params may matter at 20B)
- Standard architectures are well-tested and debugged

**Most Surprising Finding:** RoPE position embeddings had zero measurable impact, suggesting that Transformers can learn positional relationships implicitly for simple tasks.

---

## Appendices

### A. Validation Loss Table (Full Precision)

| Ablation | Best Val Loss | Final Val Loss | Min Train Loss | Final Train Loss |
|----------|---------------|----------------|----------------|------------------|
| Baseline | 1.390205 | 1.392341 | 1.234845 | 1.418084 |
| Post-norm | 1.389722 | 1.391021 | 1.241731 | 1.418499 |
| NoPE | 1.390213 | 1.392411 | 1.234891 | 1.418200 |
| No Layer Norm | 1.398479 | 1.398479 | 1.252514 | 1.428341 |
| SiLU FFN | 1.410135 | 1.411587 | 1.265550 | 1.447542 |

### B. Training Configuration Details

**Data Pipeline:**
- Training data: `tinystories_train_tokens.npy` (processed with GPT-2 tokenizer)
- Validation data: `tinystories_tokens.npy` (separate split)
- Total tokens processed: 327,680,000 (40,000 iterations × 32 batch × 256 length)

**Hardware:**
- GPU: H100 (80GB)
- Mixed Precision: FP32 (for consistency across ablations)
- Compilation: torch.compile enabled (for speed)

**Evaluation:**
- Validation frequency: Every 400 iterations
- Validation samples: 20 batches
- Checkpoint frequency: Every 4,000 iterations

### C. Generated Plots

The file `ablation_comparison.png` contains:
1. Training loss curves (all ablations overlaid)
2. Validation loss curves (all ablations overlaid)
3. Learning rate schedule
4. Performance comparison bar chart

---

*Report generated from experiments run on TinyStories dataset*
*Model: FlexibleTransformerLM (22.7M parameters)*
*Framework: PyTorch 2.0+ with custom training loop*
