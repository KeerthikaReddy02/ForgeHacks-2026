from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
VISION_MODEL = MODELS / "vision.pt"
VISION_METRICS = MODELS / "vision_metrics.json"
DISASTER_MODEL = MODELS / "disaster.joblib"
DISASTER_METRICS = MODELS / "disaster_metrics.json"
