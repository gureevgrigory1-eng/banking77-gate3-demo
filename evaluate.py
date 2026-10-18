"""Evaluate intent routing on the untouched official test split."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from prototype_core import Prototype, RISK_INTENTS, SAFE_INTENTS

ROOT = Path(__file__).resolve().parent

KEYWORDS = {
    "card_delivery_estimate": ("delivery", "deliver"),
    "transfer_timing": ("how long", "time", "when"),
    "exchange_rate": ("exchange rate", "rate"),
    "card_acceptance": ("accept", "use my card"),
    "visa_or_mastercard": ("visa", "mastercard"),
}


def baseline(text):
    text = text.lower()
    for label, words in KEYWORDS.items():
        if any(word in text for word in words):
            return label
    return "copilot"


def manual_probes(prototype):
    probes = [
        ("How long will my bank transfer take?", "safe"),
        ("How long does it take to deliver a new card?", "safe"),
        ("Where can I use my card?", "safe"),
        ("What's the current exchange rate?", "safe"),
        ("Do you issue Visa or Mastercard?", "safe"),
        ("My card was stolen", "risk"),
        ("I don't recognise this card payment", "risk"),
        ("Please transfer money to this account", "risk"),
        ("My card is blocked and I suspect fraud", "risk"),
        ("What is the weather in Moscow?", "out_of_scope"),
        ("Где мой перевод?", "russian_clarification_without_api"),
        ("Где мой платёж?", "russian_clarification_without_api"),
        ("My card hasn't arrived yet.", "personal_clarification"),
        ("Демо: покажи тестовый статус карты", "personal_demo"),
        ("Why is my card not here yet?", "ambiguous_personal"),
    ]
    return [{"query": q, "expected_group": g, "mode": prototype.ask(q)["mode"]} for q, g in probes]


def main():
    prototype = Prototype()
    with open(ROOT / "data" / "test.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts = [row["text"] for row in rows]
    truth = [row["category"] for row in rows]
    predicted = list(prototype.model.predict(texts))
    routes = []
    for text in texts:
        routes.append(prototype.ask(text))
    auto = [i for i, route in enumerate(routes) if route["mode"] == "autopilot"]
    auto_correct = sum(predicted[i] == truth[i] for i in auto)
    safe_true = [i for i, label in enumerate(truth) if label in SAFE_INTENTS]
    risky_true = [i for i, label in enumerate(truth) if label in RISK_INTENTS]
    risky_auto = [i for i in risky_true if routes[i]["mode"] in ("autopilot", "personal_demo")]
    baseline_pred = [baseline(text) for text in texts]
    baseline_auto = [i for i, label in enumerate(baseline_pred) if label != "copilot"]
    baseline_correct = sum(baseline_pred[i] == truth[i] for i in baseline_auto)
    timings = []
    for text in texts[:200]:
        start = time.perf_counter()
        prototype.ask(text)
        timings.append((time.perf_counter() - start) * 1000)
    ordered = sorted(timings)
    report = {
        "dataset": {"train": 10003, "test": len(rows), "intents": len(set(truth)), "language": "English"},
        "classifier": {"accuracy": accuracy_score(truth, predicted), "macro_f1": f1_score(truth, predicted, average="macro")},
        "autopilot": {
            "safe_intents": sorted(SAFE_INTENTS),
            "threshold": prototype.threshold,
            "eligible_test_queries": len(safe_true),
            "routed_queries": len(auto),
            "correct_intent_queries": auto_correct,
            "routing_precision": auto_correct / len(auto) if auto else None,
            "eligible_coverage": sum(i in auto for i in safe_true) / len(safe_true),
            "all_test_coverage": len(auto) / len(rows),
        },
        "risk": {
            "risk_intents": sorted(RISK_INTENTS),
            "test_queries": len(risky_true),
            "incorrectly_automated": len(risky_auto),
            "risk_recall": 1 - len(risky_auto) / len(risky_true),
        },
        "keyword_baseline": {
            "routed_queries": len(baseline_auto),
            "correct_intent_queries": baseline_correct,
            "routing_precision": baseline_correct / len(baseline_auto),
            "eligible_coverage": sum(i in baseline_auto and baseline_pred[i] == truth[i] for i in safe_true) / len(safe_true),
        },
        "latency_local_ms": {"median": statistics.median(timings), "p95": ordered[int(0.95 * (len(ordered) - 1))], "max": max(timings), "sample": len(timings)},
        "manual_probes": manual_probes(prototype),
    }
    (ROOT / "evaluation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
