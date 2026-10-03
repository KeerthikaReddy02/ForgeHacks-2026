"""Download both Kaggle datasets into data/. Needs Kaggle credentials (~/.kaggle/kaggle.json
or KAGGLE_USERNAME / KAGGLE_KEY).   uv run --group train python -m backend.ml.download_data"""
from .paths import DATA

DATASETS = {
    "wildfire": "abdelghaniaaba/wildfire-prediction-dataset",
    "disaster": "devindergarg/natural-disaster-prediction-dataset",
}

if __name__ == "__main__":
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    for name, slug in DATASETS.items():
        dest = DATA / name
        print(f"Downloading {slug} -> {dest}")
        api.dataset_download_files(slug, path=str(dest), unzip=True, quiet=False)
