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
Image and disaster endpoints use the trained models in `models/` when present and fall back to mock output (`"mock": true`) when not.
```
# 1. Data (needs Kaggle credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
uv run --group train python -m backend.ml.download_data     # -> data/wildfire, data/disaster

# 2. Wildfire satellite classifier: EfficientNet-B0 fine-tune -> models/vision.pt
uv run python -m backend.ml.train_vision --epochs 4          # add --max-per-class 2000 for a quick run

# 3. Disaster classifier: auto-detects schema, picks best of LogReg / RandomForest / HistGB by CV
uv run python -m backend.ml.train_disaster                   # or --file X.csv --target COL
```
Metrics (accuracy, per-class precision/recall/F1, confusion matrix) are written to `models/*_metrics.json` and served at `GET /api/model-info`. Commit the `models/` files so Heroku ships them.

### Results
_Fill in after training on the real datasets (copy from `models/*_metrics.json`)._

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Wildfire image (EfficientNet-B0) | | | |
| Disaster type (best of 3) | | | |
