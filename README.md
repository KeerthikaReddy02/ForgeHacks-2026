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
