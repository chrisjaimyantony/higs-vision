import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_config(path="config.yaml"):
    path = Path(path).resolve()
    config = yaml.safe_load(open(path, "r"))
    project_root = path.parent
    for key in config["paths"]:
        config["paths"][key] = str(project_root / config["paths"][key])
    return config

def load_raw(config):
    path = config["paths"]["raw_data"]
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path, header=None)
    labels = df.iloc[:, config["dataset"]["label_column"]].values
    features = df.iloc[:, config["dataset"]["feature_columns_start"]:config["dataset"]["feature_columns_end"]].values
    print(f"Loaded {features.shape[0]} events, {features.shape[1]} features")
    print(f"Signal: {(labels == 1).sum()}, Background: {(labels == 0).sum()}")
    return features, labels

def subsample(features, labels, size, seed):
    print(f"Subsampling {size} events (stratified)...")
    n_total = len(labels)
    if size >= n_total:
        print("Requested size >= dataset size, using full dataset")
        return features, labels

    indices = np.arange(n_total)
    subset_idx, _ = train_test_split(   
        indices,
        train_size=size,
        stratify=labels,
        random_state=seed
    )
    subset_idx = np.sort(subset_idx)
    print(f"Subsampled to {len(subset_idx)} events")
    print(f"Signal: {(labels[subset_idx] == 1).sum()}, Background: {(labels[subset_idx] == 0).sum()}")
    return features[subset_idx], labels[subset_idx]

def split(features, labels, config):
    seed = config["seeds"]["data_split"]
    train_r = config["split"]["train_ratio"]
    val_r = config["split"]["val_ratio"]
    test_r = config["split"]["test_ratio"]

    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=(val_r + test_r),
        stratify=labels,
        random_state=seed
    )

    # Second split: val vs test
    relative_test_size = test_r / (val_r + test_r)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        stratify=y_temp,
        random_state=seed
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("Scaler fitted on training data")
    print(f"  Feature means: {scaler.mean_[:3].round(4)}... (first 3)")
    print(f"  Feature stds:  {scaler.scale_[:3].round(4)}... (first 3)")
    return scaler

def apply_scaler(scaler, X_train, X_val, X_test):
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    print("Scaler applied to train/val/test")
    return X_train, X_val, X_test

def save_processed(X_train, y_train, X_val, y_val, X_test, y_test, scaler, config):
    out_dir = Path(config["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_test.npy", y_test)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    print(f"Saved all files to {out_dir}/")
    for f in sorted(out_dir.glob("*")):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {size_mb:.1f} MB")

def load_processed(config):
    in_dir = Path(config["paths"]["processed_dir"])

    X_train = np.load(in_dir / "X_train.npy")
    y_train = np.load(in_dir / "y_train.npy")
    X_val = np.load(in_dir / "X_val.npy")
    y_val = np.load(in_dir / "y_val.npy")
    X_test = np.load(in_dir / "X_test.npy")
    y_test = np.load(in_dir / "y_test.npy")
    scaler = joblib.load(in_dir / "scaler.pkl")

    print(f"Loaded processed data from {in_dir}/")
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

