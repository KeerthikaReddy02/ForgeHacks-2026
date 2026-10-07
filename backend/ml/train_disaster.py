"""Train the natural-disaster classifier on the Kaggle Natural Disaster Prediction dataset.

The CSV schema is auto-detected (target column, numeric / categorical / free-text features), and
the best of LogisticRegression / RandomForest / HistGradientBoosting is chosen by cross-validated
macro-F1. Writes models/disaster.joblib + disaster_metrics.json.

  uv run python -m backend.ml.train_disaster
  uv run python -m backend.ml.train_disaster --file data/disaster/x.csv --target disaster_type
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .paths import DATA, DISASTER_METRICS, DISASTER_MODEL, MODELS

ID_NAME = re.compile(r"^(id|index|unnamed.*)$|_id$", re.I)
TARGET_HINT = re.compile(r"disaster|type|label|class|target|event|category", re.I)


def load_csv(path: str | None) -> tuple[pd.DataFrame, Path]:
    if path:
        p = Path(path)
    else:
        csvs = sorted((DATA / "disaster").rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if not csvs:
            raise SystemExit(f"No CSV under {DATA / 'disaster'}; run backend.ml.download_data or pass --file")
        p = csvs[0]
    return pd.read_csv(p), p


def pick_target(df: pd.DataFrame, given: str | None) -> str:
    if given:
        return given
    cands = [c for c in df.columns if TARGET_HINT.search(c) and 2 <= df[c].nunique() <= 50]
    non_num = [c for c in cands if df[c].dtype == object]
    return (non_num or cands or [df.columns[-1]])[0]


def split_columns(X: pd.DataFrame):
    num, cat, text = [], [], []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_numeric_dtype(s):
            num.append(c)
        elif s.astype(str).str.len().mean() > 30 and s.nunique() > 50:
            text.append(c)
        else:
            cat.append(c)
    return num, cat, text


def build_pipeline(model, num, cat, text, dense):
    parts = []
    if num:
        parts.append(("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=3, sparse_output=not dense))]), cat))
    for c in text:
        parts.append((f"txt_{c}", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True), c))
    return Pipeline([("prep", ColumnTransformer(parts, sparse_threshold=0.0 if dense else 0.3)), ("clf", model)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--target")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cv-rows", type=int, default=20000, help="max rows used for model selection")
    a = ap.parse_args()

    df, path = load_csv(a.file)
    df = df.dropna(subset=[pick_target(df, a.target)]).reset_index(drop=True)
    target = pick_target(df, a.target)
    y = df[target].astype(str)
    X = df.drop(columns=[target])
    X = X.loc[:, [c for c in X.columns if X[c].nunique() < len(X) or X[c].dtype != object]]  # drop id-like
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.drop(columns=[c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"
                        if X[c].nunique() > 0.3 * len(X) and X[c].astype(str).str.len().mean() <= 30])  # names/dates
    X = X.drop(columns=[c for c in X.columns if ID_NAME.search(c) or (
        pd.api.types.is_integer_dtype(X[c]) and X[c].is_unique and X[c].is_monotonic_increasing)])
    keep = y.map(y.value_counts()) >= 5  # classes too rare to stratify
    X, y = X[keep], y[keep]
    num, cat, text = split_columns(X)
    print(f"file={path.name} rows={len(X)} target={target} classes={y.nunique()}")
    print(f"numeric={num}\ncategorical={cat}\ntext={text}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=a.seed)
    dense = not text
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced"),
        "random_forest": RandomForestClassifier(300, n_jobs=-1, class_weight="balanced_subsample",
                                                min_samples_leaf=2, random_state=a.seed),
    }
    if dense:
        candidates["hist_gb"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                               class_weight="balanced", random_state=a.seed)
    sub = Xtr.sample(min(len(Xtr), a.cv_rows), random_state=a.seed).index
    cv = StratifiedKFold(min(5, int(ytr.value_counts().min())), shuffle=True, random_state=a.seed)
    scores = {}
    for name, m in candidates.items():
        pipe = build_pipeline(m, num, cat, text, dense)
        scores[name] = float(cross_val_score(pipe, Xtr.loc[sub], ytr.loc[sub], cv=cv,
                                             scoring="f1_macro", n_jobs=1).mean())
        print(f"  cv macro-F1 {name}: {scores[name]:.4f}")
    best = max(scores, key=scores.get)
    pipe = build_pipeline(candidates[best], num, cat, text, dense).fit(Xtr, ytr)

    pred = pipe.predict(Xte)
    classes = list(pipe.classes_)
    p, r, f, _ = precision_recall_fscore_support(yte, pred, labels=classes, zero_division=0)
    m = {
        "model": best, "cv_macro_f1": {k: round(v, 4) for k, v in scores.items()},
        "accuracy": round(accuracy_score(yte, pred), 4),
        "per_class": {c: {"precision": round(p[i], 4), "recall": round(r[i], 4), "f1": round(f[i], 4)}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(yte, pred, labels=classes).tolist(),
        "labels": classes, "n_test": len(yte), "target": target, "file": path.name,
    }
    print(json.dumps(m, indent=2))

    pipe.fit(X, y)  # refit on everything for the shipped model
    # a column that is sometimes missing in training uses "missing" as its baseline: absence is signal
    base = {c: ((np.nan if X[c].isna().any() else float(X[c].median())) if c in num else X[c].mode().iloc[0])
            for c in num + cat}
    base.update({c: "" for c in text})
    meta = {"pipeline": pipe, "target": target, "num": num, "cat": cat, "text": text, "baseline": base,
            "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in num},
            "choices": {c: sorted(X[c].dropna().astype(str).unique().tolist())[:200] for c in cat}}
    MODELS.mkdir(exist_ok=True)
    joblib.dump(meta, DISASTER_MODEL, compress=3)
    DISASTER_METRICS.write_text(json.dumps(m, indent=2))
    print(f"saved {DISASTER_MODEL} ({DISASTER_MODEL.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
