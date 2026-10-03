"""Disaster-type classifier. MOCK: swap `predict_disaster` for a scikit-learn pipeline."""
import numpy as np

from ..schemas import ClassProb, DisasterInput, DisasterPrediction, FeatureContribution

CLASSES = ["Wildfire", "Flood", "Drought", "Hurricane", "Earthquake", "Heatwave"]


def predict_disaster(inp: DisasterInput) -> DisasterPrediction:
    # crude heuristic scores so the demo responds sensibly to inputs
    scores = {
        "Wildfire": inp.temperature_c / 40 + (100 - inp.humidity_pct) / 100 - inp.rainfall_mm / 100,
        "Flood": inp.rainfall_mm / 100 + inp.humidity_pct / 200,
        "Drought": (100 - inp.humidity_pct) / 100 - inp.rainfall_mm / 150,
        "Hurricane": inp.humidity_pct / 120 + inp.rainfall_mm / 200,
        "Earthquake": 0.15,
        "Heatwave": inp.temperature_c / 35 - inp.humidity_pct / 300,
    }
    if "fire" in inp.description.lower() or "smoke" in inp.description.lower():
        scores["Wildfire"] += 0.5
    v = np.array([max(scores[c], 0.01) for c in CLASSES]) ** 3
    p = v / v.sum()
    order = np.argsort(-p)[:3]

    feats = [
        FeatureContribution(feature="temperature_c", contribution=round(inp.temperature_c / 100, 3)),
        FeatureContribution(feature="humidity_pct", contribution=round(-inp.humidity_pct / 200, 3)),
        FeatureContribution(feature="rainfall_mm", contribution=round(-inp.rainfall_mm / 300, 3)),
    ]
    return DisasterPrediction(
        top3=[ClassProb(label=CLASSES[i], probability=round(float(p[i]), 3)) for i in order],
        top_features=sorted(feats, key=lambda f: -abs(f.contribution)),
    )
