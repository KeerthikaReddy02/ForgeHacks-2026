# 🔥 FireWatch

**Predicting Fire. Protecting Futures.**

Built for **ForgeHacks 2026** (AI + Climate track). FireWatch estimates wildfire risk from satellite imagery and local weather conditions, and shows historical wildfire events on a live map.

## Problem statement

- Wildfires consume over **350 million hectares** of land every year, an area larger than the European Union, causing tens of billions in economic damage.
- As the climate warms, wildfires are becoming more frequent and more intense. Despite the growing threat, the key risk data is fragmented and complex.
- Local communities and responders lack fast, easy-to-use predictive tools to assess possible risks.

This rising threat, the limited accessibility of risk assessment and the lack of historical context motivated us to build **FireWatch**: a predictive website that turns satellite imagery and climate data into immediate, actionable insights.

## Features

- **Satellite image analysis**: upload a satellite image (max 4 MB) and a ResNet-18 classifier says whether it shows wildfire or no wildfire, with a confidence score.
- **Conditions-based risk**: enter six weather and environmental factors and a LightGBM model predicts wildfire risk. All six fields are required, because the model was trained on all of them.
- **Historical risk map**: every wildfire event NASA EONET reports for a chosen year (last five years), shown live on a map.
- **Model transparency**: accuracy, precision, recall and confusion matrices are shown in the app and served at `GET /api/model-info`.

## Tech stack

| Layer | Tools |
|---|---|
| Frontend | Streamlit, pydeck |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | PyTorch / torchvision (ResNet-18), LightGBM, scikit-learn, SHAP |
| Data | Kaggle datasets, NASA EONET API |
| Tooling | uv, Make |

## Architecture

```
Streamlit UI (frontend/)  ──HTTP──▶  FastAPI (backend/main.py)
                                         ├─ services/   request handling per endpoint
                                         └─ ml/         model loading and inference
models/   saved models and metrics JSON
```

## Project layout

```
backend/
  main.py           API routes
  schemas.py        request and response models
  services/         vision, wildfire, disaster, regions, history
  ml/               inference engines and model notebooks
frontend/
  app.py            entry point, navigation, shared sidebar
  app_pages/        home, satellite, conditions, history
  assets/           logo and hero image
models/             saved models and metrics (*.json)
```

## Future scope

- **Real-time streaming and alerts**: stream live satellite feeds into an automated email/SMS alert system that notifies emergency authorities of possible new ignitions.
- **Dynamic spread modeling**: combine real-time wind, terrain and weather data to simulate and forecast 24-to-48-hour fire propagation paths.
- **Multi-source data fusion**: combine satellite imagery, live weather, vegetation dryness (NDVI), soil moisture and terrain into a single risk model instead of separate predictors.
- **Spatial risk heatmaps**: move from per-image and per-form predictions to continuous, map-based risk forecasts for any region, updated on a schedule.
- **Time-series forecasting**: predict how risk changes over the coming days from weather trends, using sequence models.
- **Fire detection and segmentation**: locate active fire fronts and burned area in an image, not just classify it.
- **Explainable predictions**: surface SHAP feature contributions and Grad-CAM heatmaps in the UI so users can see what drove each prediction.
- **Evacuation and resource planning**: suggest safe evacuation routes and where to position crews and equipment based on predicted spread.
- **Mobile and low-bandwidth access**: a mobile-friendly or offline-capable version, so remote communities and field responders can use it.
- **Public API and alert subscriptions**: let agencies and local apps pull risk data and subscribe to region-based alerts.

## Getting started

Requires Python 3.10+ (3.11 pinned) and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make dev          # backend on :8000 and frontend on :8501
```

Or run them separately:

```bash
uv run uvicorn backend.main:app --reload     # API docs at http://localhost:8000/docs
uv run streamlit run frontend/app.py         # set FIREWATCH_API to change the API URL
```

`make help` lists all targets, and `make test-api` runs a quick smoke test.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | health check |
| `POST /api/predict/image` | wildfire or no wildfire from a satellite image (multipart `file`) |
| `POST /api/predict/wildfire` | wildfire risk from weather and environmental conditions |
| `POST /api/predict/disaster` | most likely disaster type from measurements |
| `GET /api/regions/risk` | regional risk overview |
| `GET /api/regions/history?year=` | wildfire events for a year (NASA EONET) |
| `GET /api/model-info` | model metrics |

## Models

Results are on held-out test sets. The raw numbers are in `models/*_metrics.json`.

| Model | Accuracy | Details |
|---|---|---|
| Satellite image: ResNet-18, 350 px, frozen conv base | 97.1% | 6,300 test images, ROC-AUC 0.997, wildfire precision 98.7% and recall 96.1% |
| Wildfire risk from conditions: LightGBM | 96.1% | 1,835 test rows, ROC-AUC 0.985, precision 95.7%, recall 96.6% |
| Disaster type: logistic regression | 100% | 200 test rows, 4 classes (earthquake, flood, hurricane, wildfire) |

The image model was trained on raw `ToTensor()` pixels with no ImageNet normalisation, and `vision_engine.py` matches that at inference.

Datasets (Kaggle): [wildfire images](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset), [environmental factors](https://www.kaggle.com/datasets/tashasteve/quebec-wildfire-prediction-dataset). The training notebooks are in `backend/ml/`.

## Team

- Keerthika Reddy
- Harshita Singh Tanwar
- Prathit Aswar
- Dylan Fischer

## Disclaimer

Predictions are informational only and are not a substitute for official emergency alerts. In an emergency, follow your local authorities.
