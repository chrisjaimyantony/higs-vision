# HiggsVision

> Explainable and Uncertainty-Aware Deep Learning for Higgs Boson Classification

HiggsVision is a research project investigating whether deep neural networks for Higgs boson classification make **physically meaningful**, **faithful**, and **well-calibrated** decisions.

Rather than treating explainability as an afterthought, this project evaluates whether model explanations agree with established particle physics knowledge and whether those explanations genuinely reflect the model's decision-making process.

---

## Current Project Status

| Component | Status |
|-----------|--------|
| Data Pipeline | ✅ Complete |
| Exploratory Data Analysis | ✅ Complete |
| Classical ML Baselines | ✅ Complete |
| Deep Neural Network | ✅ Complete |
| Hyperparameter Optimisation (Optuna) | ✅ Complete |
| Model Evaluation | ✅ Complete |
| SHAP Explainability | 🔄 In Progress |
| Faithfulness Evaluation | 🔄 Planned |
| Physics Alignment Analysis | 🔄 Planned |
| Uncertainty Quantification | 🔄 Planned |
| Interactive Dashboard | 🔄 Planned |

---

# Motivation

Deep learning has demonstrated excellent performance on particle physics classification tasks. However, predictive performance alone is insufficient for scientific applications.

A model may achieve high accuracy while relying on spurious detector correlations or non-physical relationships. Such behaviour reduces scientific trust and limits the usefulness of machine learning in experimental physics.

This project investigates three questions:

- **Can we explain the model's decisions?**
- **Are those explanations faithful to the model?**
- **Do the explanations agree with established Higgs physics?**

---

# Dataset

**Dataset:** CERN HIGGS Dataset (Baldi et al., 2014)

- 11 million simulated collision events
- Working subset: **1,000,000 events**
- 28 physics-derived features
- Binary classification:
  - Signal
  - Background

Dataset split:

| Split | Samples |
|---------|---------:|
| Training | 700,000 |
| Validation | 150,000 |
| Test | 150,000 |

---

# Current Results

## Model Comparison

| Model | AUC | Training Time |
|------|------:|-------------:|
| Logistic Regression | 0.6850 | 3 s |
| Random Forest | 0.8201 | 343.7 s |
| XGBoost | 0.8274 | 21.8 s |
| Default DNN | 0.8452 | 18.3 min |
| **Optuna-Tuned DNN** | **0.8502** | **28.9 min** |

The tuned DNN improves upon the XGBoost baseline by approximately **0.023 AUC** while maintaining excellent generalization.

Training remained stable throughout optimization.

| Metric | Value |
|---------|------:|
| Training Loss | 0.4744 |
| Validation Loss | 0.4768 |
| Loss Gap | 0.0024 |

The minimal gap between training and validation loss indicates no evidence of significant overfitting.

---

# Best Model

Hyperparameters selected using Optuna (TPE sampler).

| Hyperparameter | Value |
|---------------|------|
| Hidden Layers | 1024 → 512 → 256 → 128 |
| Activation | GELU |
| Dropout | 0.41 |
| Weight Decay | 3.4 × 10⁻⁴ |
| Optimizer | AdamW |
| Best Validation AUC | **0.8502** |

---

# Exploratory Data Analysis

EDA was performed before model development to understand the statistical and physical characteristics of the dataset.

Key findings include:

- Successful feature standardization (mean ≈ 0, std ≈ 1)
- Balanced train/validation/test splits
- Strong correlations among invariant mass variables
- Significant overlap between signal and background distributions
- No severe outliers after preprocessing
- Physics-based hypotheses established for later SHAP validation

These observations form the baseline for later explainability experiments.

---

# Research Objectives

The remainder of the project focuses on evaluating whether the trained model behaves in a scientifically meaningful way.

## 1. SHAP Explainability

Generate both global and local explanations for the trained DNN.

Questions:

- Which physics variables drive model predictions?
- Do individual predictions rely on reasonable features?

---

## 2. Faithfulness Analysis

Evaluate whether SHAP explanations truly reflect the model's behaviour.

Planned experiments include:

- Feature masking
- Feature removal
- Model retraining
- Performance degradation analysis

---

## 3. Physics Alignment

Compare SHAP-derived feature importance against established Higgs physics expectations.

Evaluation metrics include:

- Spearman rank correlation
- Top-k overlap
- Physics group agreement

---

## 4. Uncertainty Quantification

Estimate prediction uncertainty using:

- Monte Carlo Dropout
- Deep Ensembles

The project will distinguish between:

- Aleatoric uncertainty
- Epistemic uncertainty
- Predictive uncertainty

---

# Repository Structure

```text
higgs-vision/
│
├── data/
├── notebooks/
├── src/
├── models/
├── figures/
├── results/
├── dashboard/
├── report/
├── slides/
├── config.yaml
├── requirements.txt
└── PROJECT_LOG.md
```

---

# Installation

```bash
git clone https://github.com/yourusername/higgs-vision.git

cd higgs-vision

python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt
```

Download the HIGGS dataset into

```
data/raw/
```

Run the preprocessing notebook

```
01_data_pipeline.ipynb
```

followed by the training notebook

```
03_dnn_training.ipynb
```

---

# Hardware

All current experiments were performed on

- 2019 Intel MacBook Pro
- Intel Core i9
- 16 GB RAM
- CPU-only PyTorch

The project demonstrates that competitive Higgs classification and extensive model analysis can be performed on consumer hardware without dedicated GPU resources.

---

# Roadmap

- [x] Data preprocessing
- [x] Exploratory Data Analysis
- [x] Classical baselines
- [x] Deep Neural Network
- [x] Hyperparameter optimisation
- [ ] SHAP explainability
- [ ] Faithfulness experiments
- [ ] Physics alignment analysis
- [ ] Uncertainty quantification
- [ ] Interactive dashboard
- [ ] Dissertation

---

# References

- Baldi et al. (2014) — *Searching for Exotic Particles in High-Energy Physics with Deep Learning*
- Lundberg & Lee (2017) — *SHAP*
- Gal & Ghahramani (2016) — *Dropout as Bayesian Approximation*
- Akiba et al. (2019) — *Optuna*

---

# License

This repository is released for academic and research purposes.