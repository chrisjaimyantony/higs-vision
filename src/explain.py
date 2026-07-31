import torch
import numpy as np
import shap
import json
from pathlib import Path

def compute_global_shap(model, X_train, X_test, background_seed=77, background_size=100):
    """
    Compute SHAP values for all test events using DeepExplainer.

    Args:
        model: trained HiggsDNN
        X_train: training data (for background sample)
        X_test: test data (events to explain)
        background_seed: fixed seed for reproducibility
        background_size: number of background samples

    Returns:
        shap_values: SHAP values array (same shape as X_test)
        background: the background sample used
    """
    model.eval()

    # Fixed background sample
    rng = np.random.RandomState(background_seed)
    bg_indices = rng.choice(len(X_train), size=background_size, replace=False)
    background = X_train[bg_indices]

    # Convert to tensors
    background_t = torch.FloatTensor(background)
    X_test_t = torch.FloatTensor(X_test)

    # Compute SHAP
    explainer = shap.DeepExplainer(model, background_t)
    shap_values = explainer.shap_values(X_test_t, check_additivity=False)


    # DeepExplainer returns a list; for binary with single output, take first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Flatten if needed (batch_size, 1) -> (batch_size,)
    if shap_values.ndim == 3:
        shap_values = shap_values.squeeze(-1)

    print(f"SHAP values computed: {shap_values.shape}")
    print(f"Background sample: {background_size} events (seed={background_seed})")

    return shap_values, background

def compute_feature_importance(shap_values, feature_names=None):
    """
    Rank features by mean absolute SHAP value.

    Args:
        shap_values: SHAP values array (n_events, n_features)
        feature_names: list of feature names (optional)

    Returns:
        importance: dict mapping feature name to mean |SHAP| value, sorted descending
        ranking: list of feature names in importance order
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(mean_abs_shap))]

    # Sort by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    importance = {feature_names[i]: round(float(mean_abs_shap[i]), 6) for i in sorted_indices}
    ranking = [feature_names[i] for i in sorted_indices]

    return importance, ranking

def compute_local_shap(shap_values, event_index, feature_names=None):
    """
    Get SHAP explanation for a single event.

    Args:
        shap_values: full SHAP array
        event_index: which event to explain
        feature_names: list of feature names

    Returns:
        event_shap: dict mapping feature to SHAP value for this event
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]

    event_shap = shap_values[event_index]
    sorted_indices = np.argsort(np.abs(event_shap))[::-1]

    result = {feature_names[i]: round(float(event_shap[i]), 6) for i in sorted_indices}
    return result

def save_shap_results(shap_values, importance, ranking, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.save(path / "shap_values.npy", shap_values)

    with open(path / "feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)

    with open(path / "feature_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2)

    print(f"SHAP results saved to {path}/")


def load_shap_results(path):
    path = Path(path)
    shap_values = np.load(path / "shap_values.npy")

    with open(path / "feature_importance.json") as f:
        importance = json.load(f)

    with open(path / "feature_ranking.json") as f:
        ranking = json.load(f)

    return shap_values, importance, ranking