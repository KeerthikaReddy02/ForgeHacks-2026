"""Regional risk map data. MOCK: replace with real model/aggregate output."""
from ..schemas import RegionRisk

_DATA = [
    ("Northern California", 40.0, -122.0, 0.82), ("Southern California", 34.2, -118.2, 0.88),
    ("Oregon", 44.0, -121.0, 0.61), ("Washington", 47.5, -120.5, 0.45),
    ("Arizona", 34.0, -111.9, 0.7), ("Colorado", 39.0, -105.5, 0.55),
    ("Texas", 31.0, -99.0, 0.4), ("Florida", 28.0, -82.0, 0.3),
    ("British Columbia", 53.7, -127.6, 0.58), ("New South Wales", -32.0, 147.0, 0.76),
    ("Victoria (AU)", -37.0, 144.5, 0.8), ("Portugal", 39.5, -8.0, 0.66),
    ("Greece", 39.0, 22.0, 0.72), ("Alberta", 54.0, -115.0, 0.5),
]


def region_risks() -> list[RegionRisk]:
    return [RegionRisk(region=r, lat=la, lon=lo, risk=k) for r, la, lo, k in _DATA]
