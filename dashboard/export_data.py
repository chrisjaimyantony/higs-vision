"""Export measured model results for the static HiggsVision experience.

Run from the project environment: python dashboard/export_data.py
No training, notebook edits, or original artifact changes are performed.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export(sample_size=256, evaluation_size=5000):
    import numpy as np
    import shap
    import torch
    from sklearn.metrics import roc_auc_score, roc_curve
    from src.models import HiggsDNN
    from src.physics_alignment import FEATURE_NAMES, PHYSICS_RANKING

    torch.set_num_threads(4)
    torch.manual_seed(42)
    checkpoint = ROOT / 'models/dnn/final_optuna/model.pt'
    info = json.loads(checkpoint.with_name('run_info.json').read_text())
    best = json.loads((ROOT / 'results/optuna/best_params.json').read_text())
    width = best['hidden_width']
    cfg = {
        'hidden_dims': [width, width // 2, width // 4, width // 8],
        'activation': best['activation'],
        'dropout': best['dropout'],
        'batch_norm': best['batch_norm'],
    }
    model = HiggsDNN(hidden_dims=cfg['hidden_dims'], activation=cfg['activation'],
                     dropout_rate=cfg['dropout'], use_batch_norm=cfg['batch_norm'])
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    train = np.load(ROOT / 'data/processed/X_train.npy', mmap_mode='r')
    test = np.load(ROOT / 'data/processed/X_test.npy', mmap_mode='r')
    labels = np.load(ROOT / 'data/processed/y_test.npy', mmap_mode='r')
    if not 28 == train.shape[1] == test.shape[1]:
        raise ValueError('Expected the 28 HIGGS features in dataset order.')
    if not 16 <= sample_size <= evaluation_size <= len(test):
        raise ValueError('Require 16 <= sample size <= evaluation size <= test size.')
    rng = np.random.RandomState(42)
    evaluation_ids = rng.choice(len(test), evaluation_size, replace=False)
    x_eval = np.array(test[evaluation_ids], dtype=np.float32)
    y_eval = np.array(labels[evaluation_ids], dtype=int)
    ids = evaluation_ids[:sample_size]
    x = x_eval[:sample_size]
    bg_ids = np.random.RandomState(77).choice(len(train), 100, replace=False)
    background = torch.tensor(np.array(train[bg_ids], dtype=np.float32))

    def predict(values):
        with torch.no_grad():
            return np.concatenate([model(torch.tensor(values[i:i+1024])).numpy().ravel()
                                   for i in range(0, len(values), 1024)])

    predictions = predict(x_eval)
    if cfg['activation'] == 'GELU':
        from shap.explainers._deep import deep_pytorch
        deep_pytorch.op_handler['GELU'] = deep_pytorch.nonlinear_1d
    explainer = shap.DeepExplainer(model, background)
    print(f'Explaining {sample_size} held-out events against 100 training references...', flush=True)
    values = explainer.shap_values(torch.tensor(x), check_additivity=True)
    if isinstance(values, list):
        values = values[0]
    values = np.asarray(values).reshape(sample_size, 28)
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    residuals = predictions[:sample_size] - base - values.sum(axis=1)
    if np.max(np.abs(residuals)) > 0.01:
        raise ValueError('SHAP additivity check failed; refusing to publish explanations.')
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(-importance)
    k_values = [0, 1, 2, 3, 5, 7, 10, 14, 18, 22, 28]
    random_orders = [np.random.RandomState(seed).permutation(28) for seed in (100, 200, 300)]
    ablation = []
    for k in k_values:
        masked = x_eval.copy()
        masked[:, order[:k]] = 0
        shap_auc = float(roc_auc_score(y_eval, predict(masked)))
        control = []
        for random_order in random_orders:
            masked = x_eval.copy()
            masked[:, random_order[:k]] = 0
            control.append(float(roc_auc_score(y_eval, predict(masked))))
        ablation.append(dict(k=k, shap=shap_auc, random=float(np.mean(control)),
                             randomStd=float(np.std(control))))
        print(f'Masking k={k:2}: SHAP {shap_auc:.4f}, random {np.mean(control):.4f}', flush=True)
    fpr, tpr, thresholds = roc_curve(y_eval, predictions)
    roc_ids = np.unique(np.linspace(0, len(fpr)-1, 120).astype(int))
    hist = []
    for low in np.linspace(0, 0.95, 20):
        selected = (predictions >= low) & (predictions < low + 0.05000001)
        hist.append(dict(x=float(low), signal=int(np.sum(selected & (y_eval == 1))),
                         background=int(np.sum(selected & (y_eval == 0)))))
    from scipy.stats import spearmanr
    model_ranks = {FEATURE_NAMES[i]: rank for rank, i in enumerate(order)}
    rho = float(spearmanr(range(28), [model_ranks[f] for f in PHYSICS_RANKING]).statistic)
    output = {
        'schemaVersion': 1,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'provenance': {
            'kind': 'measured', 'checkpoint': 'models/dnn/final_optuna/model.pt',
            'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'method': 'SHAP DeepExplainer', 'outputSpace': 'probability',
            'backgroundSize': 100, 'backgroundSeed': 77, 'samplingSeed': 42,
            'explanationSize': sample_size, 'evaluationSize': evaluation_size,
            'maxAdditivityResidual': float(np.max(np.abs(residuals))),
            'featureUnits': 'Training-standardized z-scores, not GeV',
            'selection': 'Seeded random sample without replacement from the held-out test split',
            'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'shap': shap.__version__},
        },
        'dataset': {'train': len(train), 'validation': 150000, 'test': len(test), 'features': 28},
        'model': {'architecture': cfg['hidden_dims'], 'activation': cfg['activation'],
                  'batchNorm': cfg['batch_norm'], 'dropout': cfg['dropout'],
                  'validationAuc': info['best_val_auc'],
                  'sampleTestAuc': float(roc_auc_score(y_eval, predictions)),
                  'history': info['history']['val_auc']},
        'baseValue': base,
        'features': [dict(name=name, importance=float(importance[i]),
                          group='Reconstructed mass' if i >= 21 else 'Detector observable',
                          physicsRank=PHYSICS_RANKING.index(name)+1) for i, name in enumerate(FEATURE_NAMES)],
        'events': [dict(id=int(ids[i]), label=int(y_eval[i]), probability=float(predictions[i]),
                        values=[float(v) for v in x[i]], shap=[float(v) for v in values[i]],
                        residual=float(residuals[i])) for i in range(sample_size)],
        'faithfulness': ablation,
        'alignment': {'spearman': rho, 'massShare': float(importance[21:].sum()/importance.sum()),
                      'top5Overlap': len(set(PHYSICS_RANKING[:5]) & {FEATURE_NAMES[i] for i in order[:5]})},
        'roc': [{'fpr': float(fpr[i]), 'tpr': float(tpr[i])} for i in roc_ids],
        'histogram': hist,
    }
    destination = Path(__file__).parent / 'data/research.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(f'Exported {destination} ({destination.stat().st_size:,} bytes)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--evaluation-size', type=int, default=5000)
    args = parser.parse_args()
    export(args.samples, args.evaluation_size)
