"""Train on the official Banking77 train split, tune routing on internal validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from prototype_core import RISK_INTENTS, SAFE_INTENTS

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["text"] for row in rows], [row["category"] for row in rows]


def choose_threshold(probs, truth, classes):
    """Pick highest coverage with a 98% validation precision safety buffer."""
    top_idx = np.argmax(probs, axis=1)
    predicted = np.asarray(classes)[top_idx]
    confidence = np.max(probs, axis=1)
    truth = np.asarray(truth)
    best = (0, 0.99)
    candidates = np.arange(0.05, 1.0, 0.01)
    for threshold in candidates:
        auto = np.isin(predicted, list(SAFE_INTENTS)) & (confidence >= threshold)
        count = int(auto.sum())
        if count == 0:
            continue
        precision = float((predicted[auto] == truth[auto]).mean())
        risk_misses = int(np.isin(truth[auto], list(RISK_INTENTS)).sum())
        if precision >= 0.98 and risk_misses == 0 and count > best[0]:
            best = (count, float(threshold))
    return round(best[1], 2)


def main():
    x, y = read_rows(ROOT / "data" / "train.csv")
    x_fit, x_val, y_fit, y_val = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=45000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, max_features=60000)),
    ])
    model = Pipeline([
        ("tfidf", features),
        ("clf", LogisticRegression(C=6, max_iter=500, solver="lbfgs")),
    ])
    model.fit(x_fit, y_fit)
    probabilities = model.predict_proba(x_val)
    pred = model.classes_[probabilities.argmax(axis=1)]
    threshold = choose_threshold(probabilities, y_val, model.classes_)
    metadata = {
        "model": model,
        "threshold": threshold,
        "personal_threshold": 0.70,
        "training_examples": len(x_fit),
        "validation_examples": len(x_val),
    }
    joblib.dump(metadata, ROOT / "model.joblib", compress=3)
    report = {
        "train_split": len(x_fit),
        "validation_split": len(x_val),
        "validation_accuracy": accuracy_score(y_val, pred),
        "validation_macro_f1": f1_score(y_val, pred, average="macro"),
        "autopilot_threshold": threshold,
        "personal_threshold": 0.70,
    }
    (ROOT / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
