# Research Log
**Project:** Explainable AI for Higgs Boson Classification using Deep Neural Networks and SHAP  
**Date:** July 28, 2026  
**Phase:** Phase 1 – Exploratory Data Analysis (EDA)

---

# Objective

The objective of today's work was to perform a comprehensive exploratory data analysis (EDA) of the HIGGS dataset prior to model development. The analysis focused on understanding the statistical properties of the dataset, verifying preprocessing steps, identifying relationships between variables, and establishing hypotheses that will later be evaluated using SHAP explanations and faithfulness analysis.

This log also serves as a reference for later project phases, particularly:

- **Phase 4:** Physics alignment using SHAP feature importance.
- **Phase 4:** Faithfulness ablation experiments.
- **Phase 6:** Dataset description and analysis for the dissertation.

---

# Dataset Overview

## Dataset

- **Dataset:** HIGGS
- **Total Samples:** 1,000,000
- **Input Features:** 28 physics-derived variables
- **Target Classes:**
  - Background (0)
  - Signal (1)

The dataset contains reconstructed kinematic properties of proton-proton collision events collected for Higgs boson classification.

---

## Dataset Split

| Split | Samples |
|--------|---------:|
| Training | 700,000 |
| Validation | 150,000 |
| Testing | 150,000 |

---

## Class Distribution

| Dataset | Background | Signal |
|----------|-----------:|-------:|
| Training | 329,056 | 370,944 |
| Validation | 70,512 | 79,488 |
| Testing | 70,512 | 79,488 |

### Observation

- Approximately **47%** of the samples belong to the Background class.
- Approximately **53%** belong to the Signal class.
- The class proportions remain identical across training, validation, and testing datasets, indicating that a **stratified train-validation-test split** was performed.

A nearly balanced dataset minimizes class imbalance bias during model training and evaluation.

---

Baseline Results (Test Set, 1M subsample):
- Logistic Regression: ROC-AUC 0.685, 3.0s
- Random Forest (500 trees): ROC-AUC 0.820, 343.7s
- XGBoost (500 rounds): ROC-AUC 0.827, 21.8s
- DNN (default config, 60 epochs): ROC-AUC ~0.845, 1100.4s

Observation: DNN outperforms all baselines. XGBoost is the strongest
tree-based model. RF is slow relative to its performance gain over LogReg.

---
# Feature Preprocessing

All numerical features were standardized using **Z-score normalization**.

\[
z=\frac{x-\mu}{\sigma}
\]

where

- \(x\) = original feature value
- \(\mu\) = feature mean
- \(\sigma\) = feature standard deviation

---

## Verification of Scaling

The feature statistics after preprocessing show:

- Mean ≈ 0
- Standard Deviation ≈ 1

The scaling plot confirms that every feature has been standardized successfully.

### Conclusion

Standardization ensures that all input variables contribute on comparable numerical scales during neural network optimization and prevents features with large numerical ranges from dominating gradient updates.

---

# Feature Distribution Analysis

Histograms comparing Signal and Background events were examined for every feature.

An important observation is that **no single feature perfectly separates Signal from Background**. Instead, most variables exhibit substantial overlap, indicating that classification will rely on combinations of multiple weakly informative features.

---

## Relatively Strong Discriminative Features

The following variables exhibit the largest observable differences between Signal and Background distributions.

- `m_bb`
- `m_wbb`
- `m_wwbb`
- `missing_energy_mag`
- `jet1_pT`
- `jet2_pT`
- `jet3_pT`
- `jet1_b-tag`
- `jet2_b-tag`
- `jet3_b-tag`
- `jet4_b-tag`

### Physical Interpretation

These variables are directly related to reconstructed Higgs decay products.

- Invariant mass variables reconstruct combinations of decay products.
- Missing transverse energy captures escaping neutrinos.
- High transverse momentum jets are characteristic of energetic collision events.
- b-tag variables identify jets likely originating from b-quarks, which are strongly associated with the dominant Higgs decay channel:

\[
H \rightarrow b\bar{b}
\]

---

## Moderately Informative Features

The following variables exhibit only subtle differences between classes.

- `lepton_pT`
- `m_jj`
- `m_jjj`
- `jet4_pT`
- all `jet_eta` variables

These variables are expected to contribute primarily through nonlinear interactions rather than individually.

---

## Weakly Informative Features

The following variables display nearly identical distributions between Signal and Background.

- `lepton_phi`
- `jet1_phi`
- `jet2_phi`
- `jet3_phi`
- `jet4_phi`
- `missing_energy_phi`

### Physical Interpretation

Azimuthal angle (φ) is approximately uniformly distributed because proton-proton collisions are rotationally symmetric around the beam axis.

Consequently, these variables are not expected to be highly informative individually.

---

# Distribution Characteristics

Several common statistical distributions were observed throughout the dataset.

## Right-Skewed Features

The following variables contain many low-valued observations and relatively few high-valued events.

- lepton_pT
- jet_pT variables
- missing_energy_mag
- invariant mass variables

---

## Approximately Gaussian Features

These variables exhibit symmetric bell-shaped distributions centered around zero.

- lepton_eta
- jet_eta variables

---

## Uniform Features

The following variables are approximately uniformly distributed.

- lepton_phi
- jet_phi variables
- missing_energy_phi

This agrees with expected detector symmetry.

---

## Discrete Features

The four b-tag variables contain only three distinct values.

| Feature | Unique Values |
|----------|--------------:|
| jet1_b-tag | 3 |
| jet2_b-tag | 3 |
| jet3_b-tag | 3 |
| jet4_b-tag | 3 |

These variables represent discrete detector outputs indicating the likelihood that a jet originated from a b-quark.

Although StandardScaler was applied uniformly across all features, scaling only changes the numerical representation of these discrete states and does not alter the underlying categorical information.

---

# Correlation Analysis

A Pearson correlation matrix was computed using the training dataset.

Most feature pairs exhibit weak correlations, indicating that many variables provide complementary information.

However, several invariant mass variables show strong positive correlations.

---

## Strongest Correlated Feature Pairs

| Feature Pair | Pearson Correlation (r) |
|---------------|-----------------------:|
| m_wbb ↔ m_wwbb | **0.896** |
| m_jj ↔ m_jjj | **0.798** |
| m_jjj ↔ m_wbb | **0.615** |
| m_jjj ↔ m_wwbb | **0.589** |
| m_jlv ↔ m_wbb | **0.568** |
| m_bb ↔ m_wbb | **0.555** |
| m_jlv ↔ m_wwbb | **0.549** |

### Interpretation

These correlations arise because multiple invariant mass variables are reconstructed using overlapping combinations of jets.

Similarly, jet momentum variables exhibit moderate positive correlations because energetic collisions often produce several energetic jets simultaneously.

The phi variables exhibit minimal correlation, indicating that they contribute largely independent information.

---

# Outlier Assessment

The 1st and 99th percentiles of every standardized feature were examined to identify potential extreme values.

### Observation

No feature exceeded ±5 standard deviations after scaling.

### Conclusion

- No severe outliers were detected.
- StandardScaler appears sufficient for preprocessing.
- Neural network optimization is unlikely to be dominated by a small number of extreme observations.

---

# Expected Physics-Based Feature Importance

Before model training, the following ranking is proposed based on known Higgs decay physics together with the observed feature distributions.

| Expected Rank | Feature | Reason |
|--------------:|---------|--------|
| 1 | m_wwbb | Reconstructs the complete decay system |
| 2 | m_wbb | Strong Higgs-related invariant mass |
| 3 | m_bb | Direct reconstruction of the b-quark pair |
| 4 | missing_energy_mag | Captures escaping neutrinos |
| 5 | jet1_pT | Leading jet energy |
| 6 | jet2_pT | Secondary energetic jet |
| 7 | jet3_pT | Additional jet activity |
| 8 | jet b-tag variables | Identification of b-quark jets |
| 9 | m_jjj | Three-jet invariant mass |
| 10 | m_jj | Dijet invariant mass |

This ranking represents a **physics hypothesis** rather than an experimental result.

It will later be compared against SHAP feature importance scores.

---

# Research Hypotheses

Based on today's exploratory analysis, the following hypotheses have been established.

### H1

Invariant mass variables (`m_bb`, `m_wbb`, `m_wwbb`) will dominate SHAP feature importance.

---

### H2

Angular variables (`phi` features) will consistently receive low SHAP importance due to their nearly uniform distributions.

---

### H3

Removing highly correlated invariant mass variables during faithfulness ablation will result in relatively small decreases in predictive performance because correlated proxy variables remain available.

---

### H4

No individual feature is expected to perfectly distinguish Signal from Background.

Instead, the neural network will learn nonlinear interactions among multiple weakly discriminative physics variables.

---

# Implications for Future Project Phases

## Phase 4 – Physics Alignment

The observed feature distributions establish an expected hierarchy of feature importance.

After training, SHAP explanations will be compared against this physics-based expectation.

Agreement would indicate that the neural network has learned physically meaningful relationships.

Disagreement may reveal unexpected model behavior or previously overlooked interactions.

---

## Phase 4 – Faithfulness Ablation

The correlation analysis identifies several groups of redundant variables.

Most notably:

- m_bb
- m_wbb
- m_wwbb
- m_jjj
- m_jj

Removing one member of these groups may not substantially reduce predictive performance because correlated proxy variables remain available.

This provides an expected explanation for future ablation results.

---

## Phase 6 – Dissertation

Today's observations provide the foundation for the dataset description chapter.

Topics already documented include:

- Dataset composition
- Class balance
- Feature preprocessing
- Distribution analysis
- Correlation analysis
- Outlier assessment
- Physics interpretation
- Expected feature hierarchy
- Research hypotheses

These notes will later be incorporated into the methodology and results chapters.

---

# Summary

Today's work established a detailed statistical and physical understanding of the HIGGS dataset prior to model development.

The principal findings are:

- Successfully verified feature standardization using Z-score normalization.
- Confirmed balanced and stratified training, validation, and testing datasets.
- Identified invariant mass variables, missing transverse energy, jet momentum, and b-tag variables as the most promising discriminative features.
- Verified that angular (φ) variables provide comparatively little discriminative information.
- Quantified the strongest correlations among invariant mass variables, highlighting potential redundancy for future faithfulness experiments.
- Confirmed the absence of severe outliers after preprocessing.
- Established a physics-based hypothesis for expected SHAP feature importance.
- Defined research hypotheses that will be tested during explainability and faithfulness analysis.

This exploratory analysis provides the statistical foundation for the subsequent stages of model development, explainability evaluation, and dissertation writing.