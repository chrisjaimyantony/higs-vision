import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
import json
from pathlib import Path

def mc_dropout_predict(model, X, T=100, batch_size=4096):
    """
    Run T forward passes with dropout active.

    Args:
        model: trained HiggsDNN
        X: input features (numpy array)
        T: number of forward passes
        batch_size: inference batch size

    Returns:
        mean_pred: mean prediction across T passes (shape: n_events)
        std_pred: standard deviation across T passes (shape: n_events)
        all_preds: all T predictions (shape: T x n_events)
    """
    model.train()  # keep dropout active
    X_tensor = torch.FloatTensor(X)
    n = len(X)

    all_preds = []
    for t in range(T):
        batch_preds = []
        for i in range(0, n, batch_size):
            batch = X_tensor[i:i+batch_size]
            with torch.no_grad():
                pred = model(batch).squeeze().numpy()
            batch_preds.append(pred)
        all_preds.append(np.concatenate(batch_preds))

    all_preds = np.array(all_preds)  # shape: (T, n_events)
    mean_pred = all_preds.mean(axis=0)
    std_pred = all_preds.std(axis=0)

    model.eval()
    return mean_pred, std_pred, all_preds

def ensemble_predict(models, X, T=100, batch_size=4096):
    """
    Run MC Dropout on each ensemble member.

    Args:
        models: list of trained HiggsDNN instances
        X: input features (numpy array)
        T: number of MC Dropout passes per model
        batch_size: inference batch size

    Returns:
        ensemble_mean: mean prediction across all models and passes
        ensemble_std: total standard deviation
        per_model_means: mean prediction per model (shape: n_models x n_events)
        per_model_stds: MC dropout std per model
        aleatoric: aleatoric uncertainty per event
        epistemic: epistemic uncertainty per event
    """
    per_model_means = []
    per_model_stds = []

    for i, model in enumerate(models):
        print(f"  Running MC Dropout on model {i+1}/{len(models)}...")
        mean_pred, std_pred, _ = mc_dropout_predict(model, X, T=T, batch_size=batch_size)
        per_model_means.append(mean_pred)
        per_model_stds.append(std_pred)

    per_model_means = np.array(per_model_means)  # (n_models, n_events)
    per_model_stds = np.array(per_model_stds)

    # Ensemble mean: average of all model means
    ensemble_mean = per_model_means.mean(axis=0)

    # Aleatoric: average within-model variance
    aleatoric = (per_model_stds ** 2).mean(axis=0)  # average variance
    aleatoric = np.sqrt(aleatoric)  # convert back to std

    # Epistemic: variance of model means
    epistemic = per_model_means.std(axis=0)

    # Total uncertainty
    ensemble_std = np.sqrt(aleatoric**2 + epistemic**2)

    return ensemble_mean, ensemble_std, per_model_means, per_model_stds, aleatoric, epistemic


def compute_ece(y_true, y_pred_proba, n_bins=15):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred_proba[mask].mean()
        bin_count = mask.sum()
        bin_weight = bin_count / len(y_true)
        gap = abs(bin_acc - bin_conf)
        ece += bin_weight * gap
        bin_data.append({
            "bin_lower": round(bin_edges[i], 3),
            "bin_upper": round(bin_edges[i + 1], 3),
            "count": int(bin_count),
            "accuracy": round(float(bin_acc), 4),
            "confidence": round(float(bin_conf), 4),
            "gap": round(float(gap), 4)
        })

    return round(float(ece), 4), bin_data

def test_uncertainty_separates_errors(y_true, y_pred, epistemic_uncertainty):
    """
    Test whether misclassified events have higher epistemic uncertainty.

    Returns:
        result: dict with test statistic, p-value, and group means
    """
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_unc = epistemic_uncertainty[correct_mask]
    incorrect_unc = epistemic_uncertainty[incorrect_mask]

    # Mann-Whitney U test (non-parametric, doesn't assume normality)
    statistic, p_value = stats.mannwhitneyu(
        incorrect_unc, correct_unc, alternative='greater'
    )

    result = {
        "test": "Mann-Whitney U (one-sided)",
        "hypothesis": "Misclassified events have higher epistemic uncertainty",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_001": p_value < 0.01,
        "mean_epistemic_correct": float(correct_unc.mean()),
        "mean_epistemic_incorrect": float(incorrect_unc.mean()),
        "median_epistemic_correct": float(np.median(correct_unc)),
        "median_epistemic_incorrect": float(np.median(incorrect_unc)),
        "n_correct": int(correct_mask.sum()),
        "n_incorrect": int(incorrect_mask.sum())
    }

    return result

def test_uncertainty_separates_errors(y_true, y_pred, epistemic_uncertainty):
    """
    Test whether misclassified events have higher epistemic uncertainty.

    Returns:
        result: dict with test statistic, p-value, and group means
    """
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask

    correct_unc = epistemic_uncertainty[correct_mask]
    incorrect_unc = epistemic_uncertainty[incorrect_mask]

    # Mann-Whitney U test (non-parametric, doesn't assume normality)
    statistic, p_value = stats.mannwhitneyu(
        incorrect_unc, correct_unc, alternative='greater'
    )

    result = {
        "test": "Mann-Whitney U (one-sided)",
        "hypothesis": "Misclassified events have higher epistemic uncertainty",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_001": p_value < 0.01,
        "mean_epistemic_correct": float(correct_unc.mean()),
        "mean_epistemic_incorrect": float(incorrect_unc.mean()),
        "median_epistemic_correct": float(np.median(correct_unc)),
        "median_epistemic_incorrect": float(np.median(incorrect_unc)),
        "n_correct": int(correct_mask.sum()),
        "n_incorrect": int(incorrect_mask.sum())
    }

    return result

def save_uncertainty_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **results)
    print(f"Uncertainty results saved to {path}")


def load_uncertainty_results(path):
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}