# FireWatch

Wildfire risk app: FastAPI backend + Streamlit frontend. Image and disaster endpoints use trained models (see *Train the models*); the preparedness and map endpoints are still mocks in `backend/services/`.

## Run
```
uv sync
uv run uvicorn backend.main:app --reload     # :8000, docs at /docs
uv run streamlit run frontend/app.py         # :8501 (FIREWATCH_API to override URL)
```

## API
| Endpoint | Swap-in |
|---|---|
| `POST /api/predict/image` (multipart `file`) | `services/vision.py` → PyTorch CNN + Grad-CAM |
| `POST /api/predict/disaster` | `services/disaster.py` → scikit-learn model |
| `POST /api/preparedness` | `services/preparedness.py` → LLM |
| `GET /api/regions/risk` | `services/regions.py` |

*Predictions are informational and not a substitute for official emergency alerts.*

## Deploy to Heroku
One web dyno runs both processes (`start.sh`): Streamlit on `$PORT`, FastAPI internally on `127.0.0.1:8000`. The Python buildpack installs from `pyproject.toml` + `uv.lock`; `.python-version` pins 3.11.
```
heroku login
heroku create <app-name>
heroku stack:set heroku-24
git push heroku feat/init-scaffold:main     # or: git push heroku main
heroku open
heroku logs --tail
```
Later: set LLM/model secrets with `heroku config:set KEY=value`. Large model files (PyTorch) can exceed Heroku's 500 MB slug limit; use CPU-only torch or load weights from external storage.

## Train the models
Image and disaster endpoints use the trained models when present (`models/vision.pt` from `train_vision`, otherwise the notebook's ResNet-18 `.pth`; `models/disaster.joblib`) and fall back to mock output (`"mock": true`) when not.
```
# 1. Data (public, no Kaggle login needed)
uv run python -m backend.ml.download_data     # -> data/wildfire, data/disaster

# 2. Wildfire satellite classifier: EfficientNet-B0 fine-tune -> models/vision.pt
uv run python -m backend.ml.train_vision --epochs 4          # add --max-per-class 2000 for a quick run

# 3. Disaster classifier: auto-detects schema, picks best of LogReg / RandomForest / HistGB by CV
uv run python -m backend.ml.train_disaster                   # or --file X.csv --target COL
```
Metrics (accuracy, per-class precision/recall/F1, confusion matrix) are written to `models/*_metrics.json` and served at `GET /api/model-info`. Commit the `models/` files so Heroku ships them.

### Results
Held-out test sets. Raw numbers: `models/*_metrics.json` (also at `GET /api/model-info`).

| Model | Accuracy | Notes |
|---|---|---|
| Wildfire image: ResNet-18, 350 px (`backend/ml/satellite_image_classification/`) | 99.27% | 6300 test images, ROC-AUC 0.9997; wildfire precision 99.1%, recall 99.6%; confusion matrix [[2788, 32], [14, 3466]] (rows = actual nowildfire, wildfire) |
| Disaster type: logreg | 100% | 200 held-out rows, 4 classes (earthquake, flood, hurricane, wildfire) |

**Read the disaster number carefully.** The Kaggle disaster dataset is synthetic: the label is fully determined by *which* measurements are present (magnitude/depth means earthquake, wind speed means hurricane, rainfall means flood, none of those means wildfire). Temperature, humidity and location alone score ~24% (chance for 4 classes). The model is therefore only meaningful when users leave unmeasured fields blank, which is how the UI works, and a request with no measurements is rejected.

**Vision model preprocessing:** the notebook trained on raw `ToTensor()` pixels with no ImageNet normalisation; inference must match (`vision_engine.py` does). The notebook's own `test_ml_model.py` normalises and so reports ~50-60% confidence on its sample images; without normalisation they score ~100%.
