# Assignment 3: Scaling Laws - Requirements

## Assignment Overview

**Goal**: Predict compute-optimal model size and hyperparameters for a FLOPs budget of 1e19.

**Key Constraints**:
- Scaling laws fitting budget: 2e18 FLOPs (hard cap enforced by API)
- Target prediction budget: 1e19 FLOPs
- Batch size must be 128 or 256 for final prediction

## Part 1/2: IsoFLOPs Method (5 points)

**Task**: Reproduce the Chinchilla IsoFLOPs scaling law fitting method.

**Input**: `data/isoflops_curves.json` - synthetic training run data
- Format: JSON array with objects containing `parameters`, `compute_budget`, `final_loss`

**Method**:
1. For each compute budget C_i, find optimal model size N_opt(C_i) that minimizes training loss
2. Fit power laws: N_opt ∝ C^a and D_opt ∝ C^b
3. Extrapolate to larger budgets (10^23 and 10^24 FLOPs)

**Deliverables**:
1. **Plot 1**: Scaling law for model size by compute budget
   - Show data points ⟨C_i, N_opt(C_i)⟩ used for fitting
   - Extrapolate up to at least 10^24 FLOPs
   - Include predicted optimal model sizes for 10^23 and 10^24 FLOPs

2. **Plot 2**: Scaling law for dataset size by compute budget
   - Show data points ⟨C_i, D_opt(C_i)⟩ used for fitting
   - Extrapolate up to at least 10^24 FLOPs
   - Include predicted optimal dataset sizes for 10^23 and 10^24 FLOPs

**Notes**:
- Use `scipy.optimize.curve_fit` or any curve fitting method
- For each IsoFLOP profile, take the run with lowest loss as minimum (instead of fitting quadratic)

---

## Part 2: Construct Scaling Laws (50 points)

**Task**: Use training API to construct scaling laws and predict optimal model for 1e19 FLOPs budget.

### Training API Details

**Base URL**: `http://hyperturing.stanford.edu:8000` (requires Stanford VPN)

**API Key**: Your SSH public key (no newlines)

**Endpoints**:

1. `GET /loss` - Query training loss for a configuration
   - Parameters:
     - `d_model`: int [64, 1024]
     - `num_layers`: int [2, 24]
     - `num_heads`: int [2, 16]
     - `batch_size`: int, one of {128, 256}
     - `learning_rate`: float [1e-4, 1e-3]
     - `train_flops`: int, one of {1e13, 3e13, 6e13, 1e14, 3e14, 6e14, 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18}
     - `api_key`: string
   - Returns: `{"loss": float, "total_flops_used": float}`

2. `GET /total_flops_used` - Check FLOPs used by your API key
   - Parameters: `api_key`
   - Returns: float

3. `GET /previous_runs` - Get all previous runs
   - Parameters: `api_key`
   - Returns: `{"previous_runs": [...]}`

**Important**: Re-querying the same configuration doesn't count against budget.

### Model Architecture

- Transformer language model (see `cs336_scaling/model.py`)
- Key specs:
  - Absolute position embeddings (not RoPE)
  - Layer normalization (not RMSNorm)
  - FFN: Linear → GeLU → Linear (d_ff = 4 * d_model)
  - Attention and residual dropout (p=0.1)
  - Untied input/output embeddings
  - Context length: 512
  - Vocabulary size: 32K (byte-level BPE)
  - Dataset: SlimPajama
  - Optimizer: AdamW (weight_decay=0.01, grad_clip=1.0)
  - LR schedule: Cosine decay (10× reduction, no warmup)
- Parameter estimation: `12 * num_layers * d_model^2` (non-embedding)

### Deliverables

**1. Writeup (writeup.pdf)**:
A complete typeset description addressing:
- **Run selection strategy**: How did you decide which runs to query within 2e18 FLOPs budget?
- **Scaling law method**: What concrete method(s) did you use to fit scaling laws? (e.g., IsoFLOPs, Kaplan et al., Hoffmann et al., Yang et al.)
- **Fit quality**: How well does your scaling law fit the experimental data?
- **Predictions for 1e19 FLOPs**:
  - Optimal model size (parameters)
  - Predicted training loss
  - Selected hyperparameters (d_model, num_layers, num_heads, batch_size, learning_rate)
- **Hyperparameter selection**: How did you determine hyperparameters for the predicted optimal size?
- **Design decisions**: Commentary on why you made particular choices

Description should be detailed enough to reproduce your approach.

**2. Code (code.zip)**:
All code written to:
- Query the training API
- Fit scaling laws
- Compute predictions
- Generate plots and analysis

**3. Google Form Submission**: https://forms.gle/sAUSLwCUETew2hYN6
- Predicted optimal model size (number of parameters)
- Training hyperparameters (d_model, num_layers, num_heads, batch_size ∈ {128, 256}, learning_rate)
- Predicted training loss

**Grading**: Part of your grade depends on the actual performance of your predicted model.

### Key Questions to Address

Your writeup should discuss:
1. Run selection strategy given 2e18 FLOPs budget
2. Scaling law fitting methodology
3. Quality of fit to experimental data
4. Predicted optimal model size for 1e19 FLOPs
5. Predicted loss for optimal model
6. Hyperparameter selection process for the predicted model

---

## Submission Checklist

- [ ] `writeup.pdf` - Complete methodology and results
- [ ] `code.zip` - All code for fitting and predictions
- [ ] Google form - Predicted model size, hyperparameters, and loss
- [ ] Verify batch size is 128 or 256
- [ ] Verify total API usage ≤ 2e18 FLOPs

---

## References

- Kaplan et al., 2020: "Scaling laws for neural language models" (arXiv:2001.08361)
- Hoffmann et al., 2022: "Training compute-optimal large language models" (Chinchilla, arXiv:2203.15556)
- Yang et al., 2022: "Tensor Programs V: Tuning large neural networks via zero-shot hyperparameter transfer" (arXiv:2203.03466)