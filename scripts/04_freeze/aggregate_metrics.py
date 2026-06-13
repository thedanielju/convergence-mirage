#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
MANIFEST = RESULTS_DIR / "frozen_manifest.json"
OUT_DIR = RESULTS_DIR / "aggregated"
OUT_JSON = OUT_DIR / "headline_metrics.json"
OUT_CSV = OUT_DIR / "headline_metrics.csv"
OUT_CONF = OUT_DIR / "per_model_aggregate_confusion.npz"

DEEP_MODELS = {"cnn2d", "bilstm", "transformer", "mamba1", "mamba2"}
CLASSICAL_MODELS = {"svm_rbf", "random_forest", "xgboost"}
NUM_CLASSES = 16
EPS = 1e-12


def _entry_dir(entry: dict) -> Path:
    rel = entry.get("output_dir_repo_relative")
    if rel:
        return REPO_ROOT / rel
    p = entry.get("output_dir_abs") or entry.get("copied_back_path") or entry.get("source_path")
    return Path(p)


def _load_arrays(run_dir: Path, model_key: str):
    y_pred = np.load(run_dir / "predictions.npy")
    y_true = np.load(run_dir / "test_labels.npy")
    proba = None
    if (run_dir / "probabilities.npy").exists():
        proba = np.load(run_dir / "probabilities.npy")
    elif (run_dir / "decision_scores.npy").exists():
        ds = np.load(run_dir / "decision_scores.npy")
        e = np.exp(ds - ds.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)
    return y_true.astype(int), y_pred.astype(int), proba


def _ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _per_class_roc_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int):
    out = []
    for c in range(n_classes):
        y_c = (y_true == c).astype(int)
        if y_c.sum() == 0 or y_c.sum() == len(y_c):
            out.append(None)
            continue
        try:
            out.append(float(roc_auc_score(y_c, proba[:, c])))
        except Exception:
            out.append(None)
    return out


def _compute_metrics(y_true, y_pred, proba, is_deep: bool) -> dict:
    labels = list(range(NUM_CLASSES))
    metrics: dict[str, Any] = {}
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0))
    metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
    metrics["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    metrics["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    metrics["kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    p_per, r_per, f_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    metrics["per_class_precision"] = [float(x) for x in p_per]
    metrics["per_class_recall"] = [float(x) for x in r_per]
    metrics["per_class_f1"] = [float(x) for x in f_per]

    if is_deep and proba is not None:
        metrics["top2_acc"] = float(top_k_accuracy_score(y_true, proba, k=2, labels=labels))
        metrics["top3_acc"] = float(top_k_accuracy_score(y_true, proba, k=3, labels=labels))
        clipped = np.clip(proba, EPS, 1.0)
        ent = -np.sum(clipped * np.log(clipped), axis=1)
        metrics["mean_entropy"] = float(ent.mean())
        metrics["mean_confidence"] = float(proba.max(axis=1).mean())
        metrics["ece"] = _ece(y_true, proba, n_bins=15)
        metrics["brier"] = _multiclass_brier(y_true, proba, NUM_CLASSES)
        metrics["per_class_roc_auc"] = _per_class_roc_auc(y_true, proba, NUM_CLASSES)
    return metrics


def _agg_mean_std(seed_metrics: dict[int, dict]) -> tuple[dict, dict]:
    keys = list(next(iter(seed_metrics.values())).keys())
    mean_d, std_d = {}, {}
    for k in keys:
        vals = [seed_metrics[s][k] for s in seed_metrics]
        if isinstance(vals[0], list):
            arr = np.array([[np.nan if v is None else v for v in vec] for vec in vals], dtype=float)
            mean_d[k] = [float(x) for x in np.nanmean(arr, axis=0)]
            std_d[k] = [float(x) for x in np.nanstd(arr, axis=0, ddof=1)] if len(vals) > 1 else [0.0] * arr.shape[1]
        else:
            arr = np.array(vals, dtype=float)
            mean_d[k] = float(np.nanmean(arr))
            std_d[k] = float(np.nanstd(arr, ddof=1)) if len(vals) > 1 else 0.0
    return mean_d, std_d


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_model: dict[str, dict[int, dict]] = defaultdict(dict)
    confusion_by_model: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))

    for entry in manifest["entries"]:
        if not entry.get("validation_passed", False):
            continue
        model = entry["model_key"]
        seed = int(entry["seed"])
        run_dir = _entry_dir(entry)
        y_true, y_pred, proba = _load_arrays(run_dir, model)
        is_deep = model in DEEP_MODELS
        m = _compute_metrics(y_true, y_pred, proba, is_deep=is_deep)
        by_model[model][seed] = m
        confusion_by_model[model] += confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    out: dict[str, Any] = {}
    csv_rows: list[dict] = []
    flat_keys = [
        "macro_f1", "accuracy", "weighted_f1", "balanced_acc",
        "macro_precision", "macro_recall", "kappa", "mcc",
        "top2_acc", "top3_acc", "mean_entropy", "mean_confidence", "ece", "brier",
    ]

    for model in sorted(by_model.keys()):
        seed_metrics = dict(sorted(by_model[model].items()))
        mean_d, std_d = _agg_mean_std(seed_metrics)
        out[model] = {
            "n_seeds": len(seed_metrics),
            "seeds": list(seed_metrics.keys()),
            "per_seed": {str(s): seed_metrics[s] for s in seed_metrics},
            "mean": mean_d,
            "std": std_d,
        }
        for s, m in seed_metrics.items():
            row = {"model": model, "seed": str(s)}
            for k in flat_keys:
                row[k] = m.get(k, "")
            csv_rows.append(row)
        for label, agg in (("mean", mean_d), ("std", std_d)):
            row = {"model": model, "seed": label}
            for k in flat_keys:
                row[k] = agg.get(k, "")
            csv_rows.append(row)

    OUT_JSON.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "seed", *flat_keys])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    np.savez(
        OUT_CONF,
        **{m: confusion_by_model[m] for m in confusion_by_model},
    )

    sha = hashlib.sha256(OUT_JSON.read_bytes()).hexdigest()
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_CONF}")
    print(f"sha256(headline_metrics.json) = {sha}")

    print("\nMean macro F1 +/- std (n=5 seeds):")
    print(f"{'model':<16}{'mean':>10}{'std':>10}")
    print("-" * 36)
    rows = sorted(out.keys(), key=lambda m: -out[m]["mean"]["macro_f1"])
    for m in rows:
        mu = out[m]["mean"]["macro_f1"]
        sd = out[m]["std"]["macro_f1"]
        print(f"{m:<16}{mu:>10.4f}{sd:>10.4f}")

    print("\nAnomaly check (std macro F1 > 0.05):")
    anomalies = [m for m in out if out[m]["std"]["macro_f1"] > 0.05]
    if anomalies:
        for m in anomalies:
            print(f"  {m}: std={out[m]['std']['macro_f1']:.4f}")
    else:
        print("  none")

    return 0


if __name__ == "__main__":
    sys.exit(main())
