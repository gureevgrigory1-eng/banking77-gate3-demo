"""Train a small Russian routing model on explicitly synthetic demo examples."""

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

ROOT = Path(__file__).resolve().parent
SAFE = {"card_delivery_estimate", "transfer_timing", "exchange_rate", "card_acceptance", "visa_or_mastercard"}


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    return [row["text"] for row in rows], [row["category"] for row in rows]


def make_model():
    return Pipeline([
        ("tfidf", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("clf", LogisticRegression(C=8, max_iter=1000, class_weight="balanced")),
    ])


def choose_threshold(model, x_val, y_val):
    probs = model.predict_proba(x_val)
    labels = model.classes_[probs.argmax(axis=1)]
    confidence = probs.max(axis=1)
    truth = np.asarray(y_val)
    for threshold in np.arange(0.35, 0.91, 0.01):
        auto = np.isin(labels, list(SAFE)) & (confidence >= threshold)
        if auto.sum() >= 5 and np.all(labels[auto] == truth[auto]):
            return round(float(threshold), 2)
    return 0.70


def main():
    train_x, train_y = read_tsv(ROOT / "data" / "ru_train.tsv")
    test_x, test_y = read_tsv(ROOT / "data" / "ru_test.tsv")
    x_fit, x_val, y_fit, y_val = train_test_split(
        train_x, train_y, test_size=0.25, stratify=train_y, random_state=42
    )
    calibration_model = make_model()
    calibration_model.fit(x_fit, y_fit)
    threshold = choose_threshold(calibration_model, x_val, y_val)
    model = make_model()
    model.fit(train_x, train_y)
    probs = model.predict_proba(test_x)
    predictions = model.classes_[probs.argmax(axis=1)]
    confidence = probs.max(axis=1)
    auto = np.isin(predictions, list(SAFE)) & (confidence >= threshold)
    truth = np.asarray(test_y)
    report = {
        "source": "Synthetic Russian paraphrases authored for this coursework; not translated Banking77",
        "train_examples": len(train_x), "held_out_examples": len(test_x),
        "classes": list(model.classes_), "threshold": threshold,
        "held_out_accuracy": accuracy_score(test_y, predictions),
        "held_out_macro_f1": f1_score(test_y, predictions, average="macro"),
        "auto_answers": int(auto.sum()),
        "auto_correct_intent": int(np.sum(predictions[auto] == truth[auto])),
        "risk_auto_answers": int(np.sum(truth[auto] == "risk")),
        "examples": [
            {"query": text, "expected": expected, "predicted": str(pred), "confidence": round(float(conf), 3)}
            for text, expected, pred, conf in zip(test_x, test_y, predictions, confidence)
        ],
    }
    joblib.dump({"model": model, "threshold": threshold}, ROOT / "ru_model.joblib", compress=3)
    (ROOT / "ru_evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
