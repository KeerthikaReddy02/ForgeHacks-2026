# FireWatch (mock stage)

Wildfire risk app: FastAPI backend + Streamlit frontend. All AI endpoints are **mocks** in `backend/services/`; replace each function body with the real model/LLM call (signatures stay the same).

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
