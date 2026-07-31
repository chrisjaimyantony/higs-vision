import numpy as np
from scipy import stats

FEATURE_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_mag", "missing_energy_phi",
    "jet1_pT", "jet1_eta", "jet1_phi", "jet1_b-tag",
    "jet2_pT", "jet2_eta", "jet2_phi", "jet2_b-tag",
    "jet3_pT", "jet3_eta", "jet3_phi", "jet3_b-tag",
    "jet4_pT", "jet4_eta", "jet4_phi", "jet4_b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]

# Physics ground truth ranking (from EDA)
PHYSICS_RANKING = [
    "m_wwbb",           # 1 - Complete decay system reconstruction
    "m_wbb",            # 2 - Higgs-related invariant mass
    "m_bb",             # 3 - b-quark pair reconstruction
    "missing_energy_mag", # 4 - Escaping neutrinos
    "jet1_pT",          # 5 - Leading jet energy
    "jet2_pT",          # 6 - Secondary jet energy
    "jet3_pT",          # 7 - Additional jet activity
    "jet1_b-tag",       # 8 - b-quark identification
    "jet2_b-tag",       # 9
    "jet3_b-tag",       # 10
    "jet4_b-tag",       # 11
    "m_jjj",            # 12 - Three-jet mass
    "m_jj",             # 13 - Dijet mass
    "lepton_pT",        # 14 - Lepton energy
    "m_lv",             # 15 - Lepton-neutrino mass
    "m_jlv",            # 16
    "jet4_pT",          # 17
    "lepton_eta",       # 18-28: weakly informative
    "jet1_eta",
    "jet2_eta",
    "jet3_eta",
    "jet4_eta",
    "lepton_phi",
    "jet1_phi",
    "jet2_phi",
    "jet3_phi",
    "jet4_phi",
    "missing_energy_phi"
]

# Feature groups for allocation analysis
HIGH_LEVEL_FEATURES = [
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]

LOW_LEVEL_FEATURES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_mag", "missing_energy_phi",
    "jet1_pT", "jet1_eta", "jet1_phi", "jet1_b-tag",
    "jet2_pT", "jet2_eta", "jet2_phi", "jet2_b-tag",
    "jet3_pT", "jet3_eta", "jet3_phi", "jet3_b-tag",
    "jet4_pT", "jet4_eta", "jet4_phi", "jet4_b-tag"
]


def compute_alignment(shap_importance, shap_ranking):
    """
    Compare SHAP feature importance with physics ground truth.
    """
    # Spearman rank correlation
    physics_to_idx = {f: i for i, f in enumerate(PHYSICS_RANKING)}
    shap_to_idx = {f: i for i, f in enumerate(shap_ranking)}

    common_features = [f for f in FEATURE_NAMES if f in physics_to_idx and f in shap_to_idx]

    physics_ranks = [physics_to_idx[f] for f in common_features]
    shap_ranks = [shap_to_idx[f] for f in common_features]

    spearman_corr, spearman_p = stats.spearmanr(physics_ranks, shap_ranks)

    # Top-5 overlap
    physics_top5 = set(PHYSICS_RANKING[:5])
    shap_top5 = set(shap_ranking[:5])
    top5_overlap = physics_top5.intersection(shap_top5)

    # Top-10 overlap
    physics_top10 = set(PHYSICS_RANKING[:10])
    shap_top10 = set(shap_ranking[:10])
    top10_overlap = physics_top10.intersection(shap_top10)

    # Feature group allocation
    total_shap_mass = sum(shap_importance.values())
    high_level_mass = sum(shap_importance.get(f, 0) for f in HIGH_LEVEL_FEATURES)
    low_level_mass = sum(shap_importance.get(f, 0) for f in LOW_LEVEL_FEATURES)

    high_level_pct = high_level_mass / total_shap_mass * 100
    low_level_pct = low_level_mass / total_shap_mass * 100

    result = {
        "spearman_correlation": round(float(spearman_corr), 4),
        "spearman_p_value": round(float(spearman_p), 6),
        "top5_overlap_count": len(top5_overlap),
        "top5_overlap_features": sorted(list(top5_overlap)),
        "top5_physics": sorted(list(physics_top5)),
        "top5_shap": sorted(list(shap_top5)),
        "top10_overlap_count": len(top10_overlap),
        "top10_overlap_features": sorted(list(top10_overlap)),
        "high_level_shap_pct": round(float(high_level_pct), 2),
        "low_level_shap_pct": round(float(low_level_pct), 2),
        "physics_ranking": PHYSICS_RANKING,
        "shap_ranking": shap_ranking
    }

    return result


def find_disagreements(shap_importance, shap_ranking, top_n=10):
    """
    Find features where SHAP and physics rankings disagree most.
    """
    physics_rank_map = {f: i + 1 for i, f in enumerate(PHYSICS_RANKING)}
    shap_rank_map = {f: i + 1 for i, f in enumerate(shap_ranking)}

    disagreements = []
    for feat in FEATURE_NAMES:
        p_rank = physics_rank_map.get(feat, 28)
        s_rank = shap_rank_map.get(feat, 28)
        disagreements.append({
            "feature": feat,
            "physics_rank": p_rank,
            "shap_rank": s_rank,
            "rank_difference": abs(p_rank - s_rank),
            "direction": "model ranks higher" if s_rank < p_rank else "physics ranks higher"
        })

    disagreements.sort(key=lambda x: x["rank_difference"], reverse=True)
    return disagreements[:top_n]