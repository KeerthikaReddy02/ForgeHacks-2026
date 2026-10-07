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
| `POST /api/predict/image` (multipart `file`) | `services/vision.py` → PyTorch CNN |
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

## Models
Image and disaster endpoints use the trained models when present (the notebook's ResNet-18 `.pth`, or `models/vision.pt` if you add one; `models/disaster.joblib`) and fall back to mock output (`"mock": true`) when not. Training code is not kept in the repo.

Metrics (accuracy, per-class precision/recall/F1, confusion matrix) live in `models/*_metrics.json` and are served at `GET /api/model-info`. Commit the `models/` files so Heroku ships them.

Datasets (Kaggle): [wildfire](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset), [disaster](https://www.kaggle.com/datasets/devindergarg/natural-disaster-prediction-dataset).

### Results
Held-out test sets. Raw numbers: `models/*_metrics.json` (also at `GET /api/model-info`).

| Model | Accuracy | Notes |
|---|---|---|
| Wildfire image: ResNet-18, 350 px, frozen conv base (`backend/ml/satellite_image_classification/`) | 97.46% | 6300 test images, ROC-AUC 0.997; wildfire precision 97.9%, recall 97.5%; confusion matrix [[2746, 74], [86, 3394]] (rows = actual nowildfire, wildfire) |
| Disaster type: logreg | 100% | 200 held-out rows, 4 classes (earthquake, flood, hurricane, wildfire) |

**Read the disaster number carefully.** The Kaggle disaster dataset is synthetic: the label is fully determined by *which* measurements are present (magnitude/depth means earthquake, wind speed means hurricane, rainfall means flood, none of those means wildfire). Temperature, humidity and location alone score ~24% (chance for 4 classes). The model is therefore only meaningful when users leave unmeasured fields blank, which is how the UI works, and a request with no measurements is rejected.

**Vision model preprocessing:** the notebook trained on raw `ToTensor()` pixels with no ImageNet normalisation; inference must match (`vision_engine.py` does).
