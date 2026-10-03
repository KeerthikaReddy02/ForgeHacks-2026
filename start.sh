#!/usr/bin/env bash
# Heroku exposes one port ($PORT): Streamlit takes it, FastAPI stays internal on 127.0.0.1:8000.
set -e
uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

exec streamlit run frontend/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
