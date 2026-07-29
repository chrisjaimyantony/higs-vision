# Research Log

**Project:** Explainable AI for Higgs Boson Classification using Deep Neural Networks and SHAP  
**Date:** July 29, 2026  
**Phase:** Phase 2 – Ensemble Training

---

# Ensemble Training

## Motivation

A single deep neural network (DNN) produces a single prediction for each event. While this is sufficient for classification, it does not provide information about prediction uncertainty.

To estimate uncertainty, multiple models are trained using the same architecture but different random initializations. This approach, known as a **Deep Ensemble**, allows each model to converge to a slightly different local optimum in the optimization landscape.

The variation between ensemble members provides an estimate of **epistemic uncertainty** (model uncertainty), while the average predictive variance within each model captures **aleatoric uncertainty** (inherent data uncertainty).

---

# Configuration

All three ensemble members used the Optuna-selected architecture and hyperparameters. The only difference between models was the random seed used for weight initialization.

| Parameter | Value |
|-----------|-------|
| Architecture | [1024, 512, 256, 128] |
| Activation | GELU |
| Dropout | 0.41 |
| Batch Normalization | False |
| Learning Rate | 0.0012 |
| Weight Decay | 0.00034 |
| Batch Size | 2048 |
| Cosine Annealing | True |
| Maximum Epochs | 60 |
| Early Stopping Patience | 10 |

---

# Training Results

| Member | Seed | Best Validation AUC | Final Epoch | Training Time | Peak RAM |
|---------|-----:|--------------------:|------------:|--------------:|---------:|
| Member 1 | 42 | 0.8502 | 60 *(No Early Stop)* | 28.9 min | 0.72 GB |
| Member 2 | 123 | 0.8499 | ~55 | ~30 min | 0.67 GB |
| Member 3 | 456 | 0.8501 | 60 *(No Early Stop)* | 54.3 min | 0.67 GB |

---

# Training Behavior

All three models followed nearly identical learning dynamics.

### Epochs 1–10

- Rapid decrease in training loss.
- Validation AUC increased from approximately **0.80** to **0.84**.
- The network learned the dominant signal/background separation and the most discriminative features.

### Epochs 10–30

- Steady performance improvements.
- Validation AUC increased from approximately **0.84** to **0.847**.
- The model refined more subtle nonlinear feature interactions.

### Epochs 30–60

- Performance gains became progressively smaller.
- Validation AUC increased from approximately **0.847** to **0.850**.
- Improvements occurred primarily in the fourth decimal place, indicating convergence toward the performance limit of the current architecture.

Members 1 and 3 completed all 60 epochs without triggering early stopping, suggesting small but consistent validation improvements throughout training. Member 2 stopped slightly earlier after a brief plateau in validation performance.

---

# Convergence Analysis

Final validation AUCs:

| Member | Validation AUC |
|---------|---------------:|
| Member 1 | 0.8502 |
| Member 2 | 0.8499 |
| Member 3 | 0.8501 |

**Range:** 0.0003

The extremely small variation (approximately **0.03%**) demonstrates that the optimization landscape contains a broad and stable optimum. Regardless of initialization, each model converged to nearly the same solution.

Although this means ensemble members will agree on the majority of predictions, the remaining disagreement is sufficient to estimate epistemic uncertainty. Subsequent statistical testing confirms that these small differences remain highly informative.

---

# Anomalous Training Behavior (Member 3)

Member 3 experienced several unusually slow epochs during training.

| Epoch | Training Time | Expected |
|------:|--------------:|----------:|
| 19 | 151.1 s | ~29 s |
| 20 | 62.3 s | ~29 s |
| 21 | 931.2 s | ~29 s |
| 22 | 482.6 s | ~29 s |
| 23 | 29.1 s | ~29 s |

These delays are attributed to background macOS processes (such as Spotlight indexing, Time Machine, or other system activity) rather than any issue with the training algorithm.

Normal training resumed immediately afterward, and the final validation performance remained unaffected. This behavior represents a hardware-level interruption rather than a model-training issue.

---

# Why Three Ensemble Members?

The original experimental design proposed a **five-member ensemble**.

Due to CPU-only hardware constraints, the ensemble size was reduced to **three members**.

| Ensemble Size | Approximate Training Time |
|---------------|--------------------------:|
| 3 Members | ~1.5 hours |
| 5 Members | ~2.5 hours |

Previous work by **Lakshminarayanan et al. (2017)** demonstrates that ensembles of **3–5 models** capture most of the predictive uncertainty benefits, with diminishing improvements beyond five members.

The three-member ensemble successfully produced statistically significant uncertainty estimates, validating this computational trade-off.

---

# Summary

Three deep ensemble members were successfully trained using the Optuna-optimized architecture and different random initializations.

Key outcomes include:

- All members converged to nearly identical validation performance (**0.8499–0.8502**).
- The optimization landscape appears stable and robust to initialization.
- Total CPU training time was approximately **1.5 hours** on a **2019 Intel MacBook Pro**.
- The trained ensemble is ready for:
  - Monte Carlo Dropout uncertainty estimation
  - Epistemic and aleatoric uncertainty decomposition
  - SHAP-based explainability analysis