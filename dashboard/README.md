# HiggsVision interactive explainer

This page introduces the Higgs signal-classification project before opening the SHAP research explorer. It includes a step-by-step collision schematic, a signal/background comparison, a physics glossary, and interactive views of saved model predictions and explanations.

## Open the dashboard

From the project root, run:

```powershell
python dashboard/serve.py
```

Then open <http://127.0.0.1:8501> in a browser. Keep the server running while using the page. Serving it locally is necessary because the frontend loads its research data as a JSON file; opening `index.html` directly will not load that data.

The included `dashboard/data/research.json` contains the exported predictions, SHAP values, and evaluation summaries, so you do not need to install the model dependencies to explore the page. If the research file is missing or needs to be regenerated, run this from the project root in the project's Python environment:

```powershell
python dashboard/export_data.py
```

Exporting requires the project dependencies in `requirements.txt`, the processed HIGGS data under `data/processed/`, and the saved checkpoint and run metadata under `models/dnn/final_optuna/`.

## What the page shows

- A schematic explanation of proton collisions, detector clues, and the classifier's signal score.
- A timed three-step walkthrough, animated collision tracks, and a full-screen presentation view.
- A local event explorer with the dataset label, prediction, and feature-level SHAP contributions.
- A global ranking of feature contributions with per-feature event plots.
- A masking comparison that checks how the model responds when influential features are removed.
- A simple project overview and a clear note about what simulated results can—and cannot—show.

The collision drawings are conceptual, not detector event displays. Model scores classify simulated dataset examples; they are not Higgs discovery probabilities or evidence from real collision data.
