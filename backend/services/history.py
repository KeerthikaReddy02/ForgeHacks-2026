"""Global wildfire history (NASA EONET, aggregated to a 2-degree grid per year)."""
import pandas as pd

from ..ml.paths import FIRE_HISTORY


def fire_history() -> list[dict]:
    if not FIRE_HISTORY.exists():
        return []
    return pd.read_csv(FIRE_HISTORY).to_dict("records")
