.PHONY: help install api ui dev stop data train-vision train-disaster eval-vision test-api heroku-local

PORT_API ?= 8000
PORT_UI  ?= 8501

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:  ## install dependencies
	uv sync

api:  ## run the FastAPI backend (reload)
	uv run uvicorn backend.main:app --reload --port $(PORT_API)

ui:  ## run the Streamlit frontend
	uv run streamlit run frontend/app.py --server.port $(PORT_UI)

dev:  ## run backend + frontend together (Ctrl-C stops both)
	@trap 'kill 0' INT TERM EXIT; \
	uv run uvicorn backend.main:app --port $(PORT_API) & \
	uv run streamlit run frontend/app.py --server.port $(PORT_UI) & \
	wait

stop:  ## stop any running backend/frontend
	-pkill -f "uvicorn backend.main"
	-pkill -f "streamlit run"

data:  ## download both Kaggle datasets into data/
	uv run python -m backend.ml.download_data

train-vision:  ## fine-tune EfficientNet-B0 -> models/vision.pt (EPOCHS=4)
	uv run python -m backend.ml.train_vision --epochs $(or $(EPOCHS),4)

train-disaster:  ## train the disaster classifier -> models/disaster.joblib
	uv run python -m backend.ml.train_disaster

eval-vision:  ## evaluate the active vision model on the test split -> models/vision_metrics.json
	uv run python -m backend.ml.eval_vision

test-api:  ## smoke-test the endpoints in-process
	uv run python -c "from fastapi.testclient import TestClient; from backend.main import app; c=TestClient(app); print(c.get('/api/health').json(), c.post('/api/predict/disaster', json={'rainfall_mm': 150}).json()['top3'][0])"

heroku-local:  ## run exactly what Heroku runs (Procfile web process)
	PORT=$(PORT_UI) bash start.sh
