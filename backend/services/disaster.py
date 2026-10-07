"""Disaster-type classifier: real scikit-learn engine if models/disaster.joblib exists, else mock."""
import logging

import numpy as np

from ..ml.paths import DISASTER_MODEL
from ..schemas import ClassProb, DisasterInput, DisasterPrediction, FeatureContribution

log = logging.getLogger(__name__)
CLASSES = ["Wildfire", "Flood", "Drought", "Hurricane", "Earthquake", "Heatwave"]


def mock_predict_disaster(inp: DisasterInput) -> DisasterPrediction:
    temp = 30.0 if inp.temperature_c is None else inp.temperature_c
    rain = 10.0 if inp.rainfall_mm is None else inp.rainfall_mm
    hum = 40.0 if inp.humidity_pct is None else inp.humidity_pct
    # crude heuristic scores so the demo responds sensibly to inputs
    scores = {
        "Wildfire": temp / 40 + (100 - hum) / 100 - rain / 100,
        "Flood": rain / 100 + hum / 200,
        "Drought": (100 - hum) / 100 - rain / 150,
        "Hurricane": hum / 120 + rain / 200,
        "Earthquake": 0.15,
        "Heatwave": temp / 35 - hum / 300,
    }
    if "fire" in inp.description.lower() or "smoke" in inp.description.lower():
        scores["Wildfire"] += 0.5
    v = np.array([max(scores[c], 0.01) for c in CLASSES]) ** 3
    p = v / v.sum()
    order = np.argsort(-p)[:3]

    feats = [
        FeatureContribution(feature="temperature_c", contribution=round(temp / 100, 3)),
        FeatureContribution(feature="humidity_pct", contribution=round(-hum / 200, 3)),
        FeatureContribution(feature="rainfall_mm", contribution=round(-rain / 300, 3)),
    ]
    return DisasterPrediction(
        top3=[ClassProb(label=CLASSES[i], probability=round(float(p[i]), 3)) for i in order],
        top_features=sorted(feats, key=lambda f: -abs(f.contribution)),
    )


_engine = None
_engine_tried = False


def predict_disaster(inp: DisasterInput) -> DisasterPrediction:
    global _engine, _engine_tried
    if not _engine_tried:
        _engine_tried = True
        if DISASTER_MODEL.exists():
            from ..ml.disaster_engine import DisasterEngine

            _engine = DisasterEngine()
        else:
            log.warning("%s missing, using mock disaster predictions", DISASTER_MODEL)
    return _engine.predict(inp) if _engine else mock_predict_disaster(inp)
