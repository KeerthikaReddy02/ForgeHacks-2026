"""Fine-tune EfficientNet-B0 on the Kaggle Wildfire Prediction (satellite) dataset.

Expects data/wildfire/{train,valid|val,test}/{wildfire,nowildfire}/*.jpg (or a single
class-folder root, which is split 80/10/10). Writes models/vision.pt + vision_metrics.json.

  uv run python -m backend.ml.train_vision --epochs 4
  uv run python -m backend.ml.train_vision --max-per-class 2000 --epochs 2   # quick CPU run
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from .paths import DATA, MODELS, VISION_METRICS, VISION_MODEL

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def find_root(root: Path) -> Path:
    """Descend through single-folder wrappers (Kaggle zips are often nested)."""
    for _ in range(4):
        subs = [p for p in root.iterdir() if p.is_dir()]
        names = {p.name.lower() for p in subs}
        if "train" in names or any(p.name.lower() in {"wildfire", "nowildfire"} for p in subs):
            return root
        if len(subs) == 1:
            root = subs[0]
        else:
            break
    return root


def pick(root: Path, *names):
    for p in root.iterdir():
        if p.is_dir() and p.name.lower() in names:
            return p


def limit(ds, n, seed):
    if not n:
        return ds
    rng = random.Random(seed)
    by_class = {}
    for i, (_, y) in enumerate(ds.samples):
        by_class.setdefault(y, []).append(i)
    idx = [i for v in by_class.values() for i in rng.sample(v, min(n, len(v)))]
    return Subset(ds, idx)


def build_datasets(root: Path, size, max_per_class, seed):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    tr, va, te = pick(root, "train"), pick(root, "valid", "val", "validation"), pick(root, "test")
    if tr and va:
        train = datasets.ImageFolder(tr, train_tf)
        val = datasets.ImageFolder(va, eval_tf)
        test = datasets.ImageFolder(te, eval_tf) if te else val
    else:  # single folder: stratified-ish random split
        full_tr, full_ev = datasets.ImageFolder(root, train_tf), datasets.ImageFolder(root, eval_tf)
        idx = list(range(len(full_tr)))
        random.Random(seed).shuffle(idx)
        a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
        train, val, test = Subset(full_tr, idx[:a]), Subset(full_ev, idx[a:b]), Subset(full_ev, idx[b:])
        train.classes = val.classes = test.classes = full_tr.classes
    classes = train.classes
    if max_per_class and hasattr(train, "samples"):
        train, val, test = (limit(d, max_per_class, seed) for d in (train, val, test))
        for d in (train, val, test):
            d.classes = classes
    return train, val, test, classes


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(dev)).argmax(1).cpu().tolist()
        ys += y.tolist()
    return ys, ps


def metrics(ys, ps, classes):
    p, r, f, _ = precision_recall_fscore_support(ys, ps, labels=range(len(classes)), zero_division=0)
    return {
        "accuracy": round(accuracy_score(ys, ps), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(ys, ps, labels=range(len(classes))).tolist(),
        "labels": list(classes),
        "n": len(ys),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "wildfire"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-per-class", type=int, default=0, help="subsample for quick runs")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = device()
    root = find_root(Path(a.data))
    train, val, test, classes = build_datasets(root, a.img_size, a.max_per_class, a.seed)
    print(f"device={dev} root={root} classes={classes} train/val/test={len(train)}/{len(val)}/{len(test)}")

    mk = lambda d, sh: DataLoader(d, a.batch_size, shuffle=sh, num_workers=a.workers,
                                  persistent_workers=a.workers > 0)
    tl, vl, el = mk(train, True), mk(val, False), mk(test, False)

    w = None if a.no_pretrained else models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=w)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.to(dev)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best, best_state = -1, None
    for ep in range(a.epochs):
        head_only = ep == 0 and a.epochs > 1 and w is not None  # warm up the new head first
        for p in model.features.parameters():
            p.requires_grad = not head_only
        params = [p for p in model.parameters() if p.requires_grad]
        if ep <= 1:
            opt = torch.optim.AdamW(params, lr=1e-3 if head_only else a.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, (a.epochs - ep) * len(tl)))
        model.train()
        t0, loss_sum = time.time(), 0.0
        for x, y in tl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item()
        ys, ps = evaluate(model, vl, dev)
        acc = accuracy_score(ys, ps)
        print(f"epoch {ep + 1}/{a.epochs} loss={loss_sum / len(tl):.4f} val_acc={acc:.4f} ({time.time() - t0:.0f}s)")
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ys, ps = evaluate(model, el, dev)
    m = metrics(ys, ps, classes)
    m["val_accuracy"] = round(best, 4)
    print(json.dumps(m, indent=2))

    MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "img_size": a.img_size}, VISION_MODEL)
    VISION_METRICS.write_text(json.dumps(m, indent=2))
    print(f"saved {VISION_MODEL} ({VISION_MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
