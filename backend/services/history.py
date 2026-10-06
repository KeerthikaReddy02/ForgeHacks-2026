"""Live global wildfire events from NASA EONET (free, no API key)."""
import time

import requests

URL = "https://eonet.gsfc.nasa.gov/api/v3/events"
TTL = 3600  # seconds to reuse a fetched year

_cache: dict[int, tuple[float, list[dict]]] = {}


def events_for_year(year: int) -> list[dict]:
    """Every wildfire event EONET reports for a calendar year: date, lat, lon, title."""
    hit = _cache.get(year)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    r = requests.get(URL, params={"category": "wildfires", "status": "all",
                                  "start": f"{year}-01-01", "end": f"{year}-12-31"}, timeout=120)
    r.raise_for_status()
    events = []
    for e in r.json()["events"]:
        g = e["geometry"][0]  # first record = where/when the fire started
        events.append({"date": g["date"][:10], "lat": g["coordinates"][1], "lon": g["coordinates"][0],
                       "title": e["title"]})
    _cache[year] = (time.time(), events)
    return events
