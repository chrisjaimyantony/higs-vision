import torch
import torch.nn as nn

class HiggsDNN(nn.Module):
    def __init__(self, input_dim=28, hidden_dims=None, activation="ReLU",
                 dropout_rate=0.3, use_batch_norm=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        # Map string to activation class
        activations = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU
        }
        act_cls = activations[activation]

        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

def create_model_from_config(config):
    dnn_cfg = config["dnn"]
    model = HiggsDNN(
        input_dim=config["dataset"]["n_features"],
        hidden_dims=dnn_cfg["hidden_dims"],
        activation=dnn_cfg["activation"],
        dropout_rate=dnn_cfg["dropout"],
        use_batch_norm=dnn_cfg["batch_norm"]
    )
    return model

def create_model_from_trial(trial, config):
    # Optuna will call this with different suggestions each trial
    width = trial.suggest_categorical("hidden_width", [256, 512, 1024])
    hidden_dims = [width, width // 2, width // 4, width // 8]

    model = HiggsDNN(
        input_dim=config["dataset"]["n_features"],
        hidden_dims=hidden_dims,
        dropout_rate=trial.suggest_float("dropout", 0.1, 0.5),
        activation=trial.suggest_categorical("activation", ["ReLU", "GELU", "SiLU"]),
        use_batch_norm=trial.suggest_categorical("batch_norm", [True, False])
    )
    return model

