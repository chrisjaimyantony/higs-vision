# HiggsVision explainer video

Run from the project root:

```powershell
python dashboard/video/render_explainer.py
```

The script reads `dashboard/data/research.json` and creates
`dashboard/video/HiggsVision_Explainer.mp4`. It uses the saved model score,
SHAP values, and clue hiding comparison; it does not retrain the model.

Requirements: Pillow, NumPy, and FFmpeg. The video is 70 seconds, 16:9, 1080p,
24 fps, and caption-led with no narration. Edit the scene text and drawings in
`render_explainer.py`, then run the command again to rebuild it.
