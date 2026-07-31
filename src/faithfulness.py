import torch
import numpy as np
import json
import copy
import time
from pathlib import Path

from src.models import HiggsDNN
from src.train import train
from src.evaluate import compute_metrics


FEATURE_NAMES = [
    "lepton_pT", "lepton_eta", "lepton_phi",
    "missing_energy_mag", "missing_energy_phi",
    "jet1_pT", "jet1_eta", "jet1_phi", "jet1_b-tag",
    "jet2_pT", "jet2_eta", "jet2_phi", "jet2_b-tag",
    "jet3_pT", "jet3_eta", "jet3_phi", "jet3_b-tag",
    "jet4_pT", "jet4_eta", "jet4_phi", "jet4_b-tag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"
]


def masking_proxy(model, X_test, y_test, shap_ranking, feature_names=None):
    """
    Phase A: Fast masking-based faithfulness proxy.
    No retraining. Masks top-k features at inference.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    results = []

    for k in range(1, len(feature_names) + 1):
        # Mask top-k features (set to training mean = 0 after scaling)
        X_masked = X_test.copy()
        top_k_indices = [feature_names.index(f) for f in shap_ranking[:k]]
        X_masked[:, top_k_indices] = 0

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_masked)).squeeze().numpy()

        metrics = compute_metrics(y_test, preds)
        results.append({
            "k": k,
            "removed_features": shap_ranking[:k],
            "roc_auc": metrics["roc_auc"],
            "f1": metrics["f1"]
        })

        if k <= 5 or k % 5 == 0 or k == len(feature_names):
            print(f"  k={k:2d}: ROC-AUC = {metrics['roc_auc']:.4f} "
                  f"(removed: {shap_ranking[min(k-1, len(shap_ranking)-1)]})")

    return results


def random_masking_proxy(model, X_test, y_test, feature_names=None, n_orderings=3, seed=100):
    """
    Phase A: Random feature ordering control.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    all_results = []

    for order_idx in range(n_orderings):
        rng = np.random.RandomState(seed + order_idx * 100)
        random_order = rng.permutation(feature_names).tolist()

        print(f"\n  Random ordering {order_idx + 1}/{n_orderings}")
        results = masking_proxy(model, X_test, y_test, random_order, feature_names)

        for r in results:
            r["ordering"] = order_idx
        all_results.append(results)

    # Average across orderings
    n_features = len(feature_names)
    averaged = []
    for k_idx in range(n_features):
        aucs = [all_results[o][k_idx]["roc_auc"] for o in range(n_orderings)]
        averaged.append({
            "k": k_idx + 1,
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs))
        })

    return averaged


def compute_abc(shap_results, random_results):
    """
    Compute Area Between Curves.
    """
    shap_aucs = np.array([r["roc_auc"] for r in shap_results])
    if isinstance(random_results[0], dict) and "roc_auc_mean" in random_results[0]:
        random_aucs = np.array([r["roc_auc_mean"] for r in random_results])
    else:
        random_aucs = np.array([r["roc_auc"] for r in random_results])

    min_len = min(len(shap_aucs), len(random_aucs))
    shap_aucs = shap_aucs[:min_len]
    random_aucs = random_aucs[:min_len]

    # ABC: area between the two curves
    # SHAP curve should drop faster, so random - shap should be positive
    abc = np.trapz(random_aucs - shap_aucs, dx=1)

    return float(abc)


def save_faithfulness_results(shap_proxy, random_proxy, abc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "shap_ordering_proxy": shap_proxy,
        "random_ordering_proxy": random_proxy,
        "area_between_curves": abc
    }

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Faithfulness results saved to {path}")