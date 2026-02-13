# CS336 Assignment 3: Scaling Laws - Writeup

**Student Name**: [Your Name]
**Date**: [Date]

---

## Part 1: IsoFLOPs Method (5 points)

### 1.1 Model Size Scaling Law

**Plot**: [Include plot: `results/part1_isoflops/model_size_scaling_law.png`]

**Fitted Power Law**:
- Formula: N_opt = [coefficient] × C^[exponent]
- Coefficient (a): [value]
- Exponent (b): [value]

**Predictions**:
- **For C = 10^23 FLOPs**: N_opt = [value] parameters
- **For C = 10^24 FLOPs**: N_opt = [value] parameters

### 1.2 Dataset Size Scaling Law

**Plot**: [Include plot: `results/part1_isoflops/dataset_size_scaling_law.png`]

**Fitted Power Law**:
- Formula: D_opt = [coefficient] × C^[exponent]
- Coefficient (a): [value]
- Exponent (b): [value]

**Predictions**:
- **For C = 10^23 FLOPs**: D_opt = [value] tokens
- **For C = 10^24 FLOPs**: D_opt = [value] tokens

---

## Part 2: Constructing Scaling Laws (50 points)

### 2.1 Experiment Design Strategy

#### 2.1.1 Budget Allocation

**Total Budget**: 2×10^18 FLOPs
**Target Prediction Budget**: 1×10^19 FLOPs

**Strategy Overview**:
[Describe your overall approach - e.g., IsoFLOPs method, number of compute budgets tested, number of model sizes per budget]

**Rationale**:
[Explain WHY you chose this strategy - e.g., need sufficient compute budgets for extrapolation, need multiple model sizes per budget to find optimal, etc.]

#### 2.1.2 Compute Budget Selection

**Selected Compute Budgets**:
[List the compute budgets you tested, e.g., 1e15, 3e15, 6e15, 1e16, 3e16, 6e16, 1e17, 3e17, 6e17, 1e18]

**Justification**:
[Explain why you chose these specific budgets - e.g., logarithmic spacing, covering range up to our largest available budget (1e18), sufficient for extrapolation to 1e19]

#### 2.1.3 Model Size Selection

**Models per Budget**: [Number]

**Model Size Strategy**:
[Describe how you selected different model sizes - e.g., varied d_model and num_layers to achieve different parameter counts, targeted range from small to large models at each budget]

**Design Considerations**:
[Explain design choices - e.g., balanced width vs. depth, head dimension preferences, constraints from API requirements]

#### 2.1.4 Hyperparameter Selection for Experiments

**Fixed Hyperparameters**:
- Batch size: [128 or 256]
- Learning rate: [value, e.g., 3e-4]

**Rationale**:
[Explain why you chose these values - e.g., standard batch size for good GPU utilization, learning rate based on literature/prior experience]

#### 2.1.5 Budget Utilization

**Total Configurations**: [Number]
**Estimated Cost**: [Value] FLOPs
**Actual Cost**: [Value] FLOPs
**Budget Utilization**: [Percentage]

[Brief comment on whether you fully utilized the budget and why]

### 2.2 Experimental Results

#### 2.2.1 Data Collection

**Total Runs Completed**: [Number]
**Compute Budgets Tested**: [Number]
**Average Models per Budget**: [Number]

**Loss Range Observed**: [Min] to [Max]

#### 2.2.2 IsoFLOPs Profiles

[Describe the IsoFLOPs profiles observed - e.g., for each compute budget, which model size achieved lowest loss, any unexpected patterns]

**Example Observations**:
- At C = 1e18 FLOPs: Best model had [parameters] parameters with loss [value]
- [Any interesting patterns - e.g., optimal model size increases with compute budget as expected]

### 2.3 Scaling Law Fitting

#### 2.3.1 Fitting Method

**Approach**: [e.g., IsoFLOPs method from Hoffmann et al., 2022]

**Technical Details**:
[Describe the concrete method - e.g., grouped runs by compute budget, found minimum loss run for each budget, fitted power laws in log-log space using least squares]

**Implementation**:
[Any relevant implementation details - e.g., used numpy.polyfit in log space, scipy.optimize.curve_fit, etc.]

#### 2.3.2 Fitted Scaling Laws

**Plot**: [Include plot: `results/part2_scaling_laws/scaling_laws.png`]

**Power Law Formulas**:

1. **Model Size Scaling Law**:
   - Formula: N_opt = [a] × C^[b]
   - Coefficient (a): [value]
   - Exponent (b): [value]

2. **Dataset Size Scaling Law**:
   - Formula: D_opt = [a] × C^[b]
   - Coefficient (a): [value]
   - Exponent (b): [value]

3. **Loss Scaling Law**:
   - Formula: L_opt = [a] × C^[b]
   - Coefficient (a): [value]
   - Exponent (b): [value]

#### 2.3.3 Fit Quality

**R² Scores**:
- Model Size: [value]
- Dataset Size: [value]
- Loss: [value]

**Assessment**:
[Discuss how well the power laws fit the experimental data - e.g., R² close to 1 indicates good fit, any systematic deviations, confidence in extrapolation]

**Visual Inspection**:
[Comment on the plots - e.g., data points align well with fitted curves, any outliers, extrapolation appears reasonable]

### 2.4 Predictions for Target Budget

**Target Compute Budget**: 1×10^19 FLOPs

#### 2.4.1 Scaling Law Predictions

From fitted power laws:

- **Optimal Model Size (N_opt)**: [value] parameters
- **Optimal Dataset Size (D_opt)**: [value] tokens
- **Predicted Training Loss**: [value]

**Confidence**:
[Discuss confidence in these predictions - e.g., target is 10× largest experiment, reasonable extrapolation, consistent with literature]

#### 2.4.2 Selected Hyperparameters

**Final Configuration for 1×10^19 FLOPs**:

**Architecture**:
- `d_model`: [value]
- `num_layers`: [value]
- `num_heads`: [value]
- **Estimated Parameters**: [value] (matches N_opt = [predicted value] with error of [percentage])

**Training**:
- `batch_size`: [128 or 256]
- `learning_rate`: [value]

**Head Dimension**: [value] (d_model / num_heads)

#### 2.4.3 Hyperparameter Selection Rationale

**Architecture Selection**:
[Explain how you chose d_model, num_layers, num_heads - e.g., searched for combination that closely matches predicted optimal parameters, preferred deeper vs. wider models, ensured head dimension is reasonable (64 or 128)]

**Learning Rate Selection**:
[Explain learning rate choice - e.g., analyzed experimental data to find best LR for similar model sizes, used scaling heuristic (smaller LR for larger models), based on literature recommendations]

**Batch Size Selection**:
[Explain batch size choice - e.g., chose 256 for better GPU utilization with large model, meets assignment constraint]

### 2.5 Analysis and Discussion

#### 2.5.1 Comparison with Literature

**Chinchilla Results**:
[Compare your scaling law exponents with Hoffmann et al., 2022 - e.g., they found N_opt ~ C^0.50 and D_opt ~ C^0.50, discuss similarities/differences]

**Possible Reasons for Differences**:
[If your exponents differ, discuss why - e.g., different model architecture, different dataset, smaller scale of experiments, etc.]

#### 2.5.2 Limitations and Uncertainties

**Experimental Limitations**:
- [e.g., Limited to compute budgets up to 1e18, extrapolating to 1e19 (10× larger)]
- [e.g., Fixed hyperparameters (batch size, learning rate) may not be optimal for all model sizes]
- [e.g., Limited number of model sizes tested per budget]

**Uncertainties in Prediction**:
- [e.g., Extrapolation beyond observed data always has uncertainty]
- [e.g., R² scores indicate fit quality but don't guarantee extrapolation accuracy]
- [e.g., Hyperparameters may need tuning for the final large model]

#### 2.5.3 Potential Improvements

**If Given More Budget**:
[Discuss what you would do differently with more FLOPs - e.g., test more compute budgets closer to 1e19, test more model sizes per budget, perform hyperparameter sweeps for learning rate and batch size]

**Alternative Approaches**:
[Mention other approaches you considered - e.g., Kaplan et al., method, fitting loss directly as function of N and D, different extrapolation methods]

### 2.6 Summary

**Key Findings**:
1. Fitted scaling laws from [number] experiments totaling [value] FLOPs
2. Predicted optimal model size for 1e19 FLOPs: [value] parameters
3. Selected hyperparameters: d_model=[value], num_layers=[value], num_heads=[value], batch_size=[value], learning_rate=[value]
4. Predicted training loss: [value]

**Confidence**: [High/Medium/Low] confidence in predictions based on [reasons]

---

## Appendix

### A. Experimental Data Summary

[Optional: Include table of all experimental runs with parameters, FLOPs, and loss]

### B. Code Structure

[Optional: Briefly describe your code organization]

**Modules**:
- `api_client.py`: API querying and caching
- `experiment_design.py`: Experiment strategy
- `scaling_law_fitter.py`: Power law fitting
- `hyperparameter_selector.py`: Hyperparameter selection
- `run_experiments.py`: Main orchestration

### C. Reproducibility

**To reproduce results**:
```bash
cd part2_scaling_laws
./run_part2.sh
```

**Requirements**:
- API key file: `api_key.txt`
- Stanford VPN connection
- Python packages: numpy, scipy, matplotlib, requests

---

## References

- Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). arXiv:2203.15556.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- [Any other references you used]

---

**Notes**:
- All plots are included in the `results/` directory
- Code is available in `part1_isoflops/` and `part2_scaling_laws/`
- Predictions submitted to Google form: [Link if available]
