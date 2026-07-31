Phase 4 Results (Faithfulness + Physics Alignment):

Faithfulness:
- ABC = 2.3778 (large — SHAP explanations are faithful)
- Removing top 3 SHAP features: AUC drops from 0.85 to 0.70
- Removing 3 random features: AUC drops from 0.84 to 0.82
- Conclusion: SHAP correctly identifies which features the model depends on

Physics Alignment:
- Spearman ρ = 0.873 (p < 0.001) — strong alignment
- Top-5 overlap: 4/5 (m_wwbb, m_wbb, m_bb, jet1_pT match)
- High-level features: 63% of total SHAP mass
- Model learned physically meaningful representations

Key Disagreements:
- b-tag variables ranked much lower than expected (possibly redundant with mass variables)
- lepton_pT ranked much higher than expected (possibly capturing nonlinear interactions)
- missing_energy_mag slightly lower than expected