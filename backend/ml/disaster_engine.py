"""Disaster classifier inference: maps API inputs onto the trained dataset's columns, returns top-3
classes and per-feature contributions (occlusion: probability drop when a feature is set to baseline)."""
import re

import joblib
import numpy as np
import pandas as pd

from ..schemas import ClassProb, DisasterInput, DisasterPrediction, FeatureContribution
from .paths import DISASTER_MODEL

ALIASES = {  # API field -> regex over dataset column names
    "temperature_c": re.compile(r"temp", re.I),
    "rainfall_mm": re.compile(r"rain|precip", re.I),
    "humidity_pct": re.compile(r"humid", re.I),
    "region": re.compile(r"region|country|location|state|place|area|city", re.I),
}
TEXT_HINT = re.compile(r"desc|text|comment|summary|note", re.I)


class DisasterEngine:
    def __init__(self, path=DISASTER_MODEL):
        m = joblib.load(path)
        self.pipe, self.base = m["pipeline"], m["baseline"]
        self.num, self.cat, self.text = m["num"], m["cat"], m["text"]
        self.cols = self.num + self.cat + self.text
        self.ranges = m["ranges"]

    def _map(self, inp: DisasterInput) -> dict:
        """dataset column -> user-supplied value."""
        supplied = {}
        for field, rx in ALIASES.items():
            val = getattr(inp, field)
            col = next((c for c in self.cols if rx.search(c)), None)
            if col is None or (field == "region" and not val):
                continue
            if col in self.num and field == "temperature_c":
                lo, hi = self.ranges[col]
                if lo > 150:  # column looks like Kelvin
                    val = val + 273.15
            if col in self.num and field == "region":
                continue
            supplied[col] = val
        if inp.description:
            col = next((c for c in self.text if TEXT_HINT.search(c)), self.text[0] if self.text else None)
            if col:
                supplied[col] = inp.description
        for col, val in (inp.features or {}).items():  # exact dataset columns
            if col in self.cols:
                supplied[col] = val
        return supplied

    def _frame(self, values: dict) -> pd.DataFrame:
        row = {c: values.get(c, self.base[c]) for c in self.cols}
        df = pd.DataFrame([row])
        for c in self.num:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def predict(self, inp: DisasterInput) -> DisasterPrediction:
        supplied = self._map(inp)
        probs = self.pipe.predict_proba(self._frame(supplied))[0]
        classes = self.pipe.classes_
        order = np.argsort(-probs)
        k = order[0]

        contrib = []
        for col in supplied:
            ablated = {c: v for c, v in supplied.items() if c != col}
            p0 = self.pipe.predict_proba(self._frame(ablated))[0][k]
            contrib.append(FeatureContribution(feature=col, contribution=round(float(probs[k] - p0), 4)))
        contrib.sort(key=lambda f: -abs(f.contribution))

        return DisasterPrediction(
            top3=[ClassProb(label=str(classes[i]), probability=round(float(probs[i]), 4)) for i in order[:3]],
            top_features=contrib[:5],
            mock=False,
        )
