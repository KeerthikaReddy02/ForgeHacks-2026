"""Evaluate the active vision model on the wildfire test split -> models/vision_metrics.json.

  uv run python -m backend.ml.eval_vision [--data data/wildfire] [--limit N]
Without --data it fetches the dataset through kagglehub (public, ~1.5 GB, cached).
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .paths import MODELS, VISION_METRICS, VISION_MODEL
from .train_vision import find_root, pick
from .vision_engine import VisionEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--limit", type=int, default=0, help="max images (evenly strided) for a quick check")
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()

    if a.data:
        root = find_root(Path(a.data))
    else:
        import kagglehub
        root = find_root(Path(kagglehub.dataset_download("abdelghaniaaba/wildfire-prediction-dataset")))
    test_dir = pick(root, "test")
    eng = VisionEngine()
    ds = ImageFolder(test_dir, eng.tf)
    assert ds.classes == eng.classes, f"class order mismatch: {ds.classes} vs {eng.classes}"
    if a.limit:
        ds = Subset(ds, range(0, len(ds), max(1, len(ds) // a.limit)))
    print(f"model={VISION_MODEL.name} arch={eng.arch} test_images={len(ds)}")

    ys, ps, pr = [], [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, a.batch_size, num_workers=4):
            prob = eng.model(x).softmax(1)
            ps += prob.argmax(1).tolist(); ys += y.tolist(); pr += prob[:, 1].tolist()
    p, r, f, _ = precision_recall_fscore_support(ys, ps, labels=range(len(eng.classes)), zero_division=0)
    m = {
        "model": eng.arch, "accuracy": round(accuracy_score(ys, ps), 4),
        "roc_auc": round(roc_auc_score(ys, pr), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
                      for i, c in enumerate(eng.classes)},
        "confusion_matrix": confusion_matrix(ys, ps).tolist(), "labels": list(eng.classes), "n": len(ys),
    }
    print(json.dumps(m, indent=2))
    MODELS.mkdir(exist_ok=True)
    VISION_METRICS.write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
