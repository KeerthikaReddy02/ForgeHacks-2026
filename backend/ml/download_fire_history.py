"""Download 5 years of global wildfire events from NASA EONET (free, no key) and aggregate them
to a 2-degree grid per year -> backend/data/fire_history.csv (committed, small).
   uv run python -m backend.ml.download_fire_history"""
import pandas as pd
import requests

from .paths import FIRE_HISTORY

URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
CELL = 2.0  # grid size in degrees


def main():
    r = requests.get(URL, params={"category": "wildfires", "status": "all", "days": 1825}, timeout=120)
    r.raise_for_status()
    rows = []
    for e in r.json()["events"]:
        g = e["geometry"][0]  # first record = where/when the fire started
        lon, lat = g["coordinates"][:2]
        ha = g.get("magnitudeValue") or 0
        if g.get("magnitudeUnit") == "acres":
            ha *= 0.4047
        elif g.get("magnitudeUnit") != "hectare":
            ha = 0
        rows.append({"year": int(g["date"][:4]), "lat": lat, "lon": lon, "hectares": ha})
    df = pd.DataFrame(rows)
    df["lat"] = (df.lat // CELL) * CELL + CELL / 2
    df["lon"] = (df.lon // CELL) * CELL + CELL / 2
    out = df.groupby(["year", "lat", "lon"], as_index=False).agg(events=("year", "size"), hectares=("hectares", "sum"))
    out["hectares"] = out.hectares.round(0)
    FIRE_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FIRE_HISTORY, index=False)
    print(f"{len(df)} events -> {len(out)} grid cells -> {FIRE_HISTORY}")


if __name__ == "__main__":
    main()
