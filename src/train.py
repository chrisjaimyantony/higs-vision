import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import optuna

def train(model, X_train, y_train, X_val, y_val, config, seed=42, trial=None):
    """
    Trains the DNN and returns the trained model plus training history.

    Args:
        model: HiggsDNN instance
        X_train, y_train: training data (numpy arrays)
        X_val, y_val: validation data (numpy arrays)
        config: project config dict
        seed: random seed for reproducibility
        trial: Optuna trial object (None when training final model)

    Returns:
        model: trained model
        history: dict with per-epoch metrics
    """
    import optuna
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

    batch_size = config["dnn"]["batch_size"]
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["dnn"]["learning_rate"],
        weight_decay=config["dnn"]["weight_decay"]
    )
    loss_fn = nn.BCELoss()

    scheduler = None
    if config["dnn"]["cosine_annealing"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["dnn"]["max_epochs"]
        )

    history = {"train_loss": [], "val_loss": [], "val_auc": [], "epoch_time": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["dnn"]["early_stopping_patience"]

    total_start = time.time()

    for epoch in range(config["dnn"]["max_epochs"]):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
            val_auc = roc_auc_score(y_val, val_pred.numpy())

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["epoch_time"].append(epoch_time)

        if scheduler:
            scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Report to Optuna if running HPO
        if trial is not None:
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | "
              f"Time: {epoch_time:.1f}s")

    # Load best weights
    model.load_state_dict(best_state)
    total_time = time.time() - total_start
    peak_ram = psutil.Process().memory_info().rss / 1e9

    history["total_time"] = total_time
    history["peak_ram_gb"] = peak_ram

    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Peak RAM: {peak_ram:.2f} GB")
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")

    return model, history


def save_training_run(model, history, config, save_dir, seed):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Save history as JSON
    run_info = {
        "seed": seed,
        "config": config["dnn"],
        "total_epochs": len(history["train_loss"]),
        "best_val_auc": max(history["val_auc"]),
        "best_val_loss": min(history["val_loss"]),
        "total_time_seconds": history["total_time"],
        "peak_ram_gb": history["peak_ram_gb"],
        "history": {k: [round(v, 6) for v in vals] if isinstance(vals, list) else vals
                    for k, vals in history.items()}
    }
    with open(save_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved model and run info to {save_dir}/")