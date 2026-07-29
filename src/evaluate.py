import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute all evaluation metrics.

    Args:
        y_true: true labels (0 or 1)
        y_pred_proba: predicted probabilities (between 0 and 1)
        threshold: classification threshold (default 0.5)

    Returns:
        dict of all metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics

def compute_ece(y_true, y_pred_proba, n_bins=15):
    """
    Expected Calibration Error.
    Measures how well predicted probabilities match observed frequencies.
    """
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
        ece += bin_weight * abs(bin_acc - bin_conf)
        bin_data.append({
            "bin_lower": round(bin_edges[i], 3),
            "bin_upper": round(bin_edges[i + 1], 3),
            "count": int(bin_count),
            "accuracy": round(float(bin_acc), 4),
            "confidence": round(float(bin_conf), 4),
            "gap": round(float(abs(bin_acc - bin_conf)), 4)
        })

    return round(float(ece), 4), bin_data

def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {path}")

def load_metrics(path):
    with open(path, "r") as f:
        return json.load(f)

def print_comparison_table(metrics_dict):
    """
    Print a comparison table across models.

    Args:
        metrics_dict: {"model_name": metrics_dict, ...}
    """
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"]

    print(f"{'Model':<20s} {'Accuracy':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'ROC-AUC':>8s} {'PR-AUC':>8s} {'Brier':>8s}")
    print("-" * 84)

    for name, m in metrics_dict.items():
        print(f"{name:<20s} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>8.4f} "
              f"{m['pr_auc']:>8.4f} {m['brier_score']:>8.4f}")