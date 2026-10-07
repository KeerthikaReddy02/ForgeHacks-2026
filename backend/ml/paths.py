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

WILDFIRE_DIR = ROOT / "backend" / "ml" / "wildfire_risk_prediction"
WILDFIRE_MODEL = WILDFIRE_DIR / "wildfire_lgbm.pkl"
WILDFIRE_EXPLAINER = WILDFIRE_DIR / "shap_explainer.pkl"
WILDFIRE_DATA = WILDFIRE_DIR / "wildfire_data_csv.csv"
WILDFIRE_METRICS= MODELS/"wildfire_metrics.json"