from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
_NOTEBOOK_MODEL = ROOT / "backend/ml/satellite_image_classification/satellite_image_classification_model.pth"
# models/vision.pt (from train_vision) wins; otherwise use the notebook-trained ResNet-18
VISION_MODEL = MODELS / "vision.pt" if (MODELS / "vision.pt").exists() or not _NOTEBOOK_MODEL.exists() else _NOTEBOOK_MODEL
VISION_METRICS = MODELS / "vision_metrics.json"
DISASTER_MODEL = MODELS / "disaster.joblib"
DISASTER_METRICS = MODELS / "disaster_metrics.json"
FIRE_HISTORY = ROOT / "backend/data/fire_history.csv"
