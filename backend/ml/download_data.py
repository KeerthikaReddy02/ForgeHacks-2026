"""Download both Kaggle datasets into data/ via kagglehub (public datasets, no credentials needed).
   uv run python -m backend.ml.download_data"""
import shutil
from pathlib import Path

import kagglehub

from .paths import DATA

DATASETS = {
    "wildfire": "abdelghaniaaba/wildfire-prediction-dataset",
    "disaster": "devindergarg/natural-disaster-prediction-dataset",
}

if __name__ == "__main__":
    for name, slug in DATASETS.items():
        dest = DATA / name
        print(f"Downloading {slug} -> {dest}")
        shutil.copytree(Path(kagglehub.dataset_download(slug)), dest, dirs_exist_ok=True)
