#!/usr/bin/env python3
"""Build a frozen manifest of admissible (model, seed) runs.

Walks provider-exported deep-model runs and local classical runs, then validates
each run's artifact set and numerical sanity.

Outputs:
  results/frozen_manifest.json        - admissible runs
  results/frozen_manifest_skips.json  - excluded runs with reasons
  results/frozen_manifest_audit.log   - one-page audit report
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

# ----- paths -------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
PROVIDER_EXPORTS = RESULTS_DIR / "provider_exports"
FINAL_DIR = RESULTS_DIR / "final"
SPLIT_MANIFEST = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready" / "manifest.csv"

OUT_MANIFEST = RESULTS_DIR / "frozen_manifest.json"
OUT_SKIPS = RESULTS_DIR / "frozen_manifest_skips.json"
OUT_AUDIT = RESULTS_DIR / "frozen_manifest_audit.log"

DEEP_REQUIRED = [
    "config.json", "environment.json", "debug_gates.json",
    "history.csv", "history.jsonl", "train.log", "best.pt",
    "metrics.json", "classification_report.json", "report.txt",
    "confusion_matrix.npy", "confusion_matrix_normalized.npy",
    "top_confusions.json", "predictions.npy", "probabilities.npy",
    "test_labels.npy", "test_track_ids.npy", "run_manifest.json",
]
# classical: no train.log, no best.pt, no debug_gates, no history.* ;
# adds model.joblib, grid_search_results.{csv,json}; svm has no probabilities.npy.
CLASSICAL_REQUIRED_BASE = [
    "config.json", "environment.json", "metrics.json",
    "classification_report.json", "report.txt",
    "confusion_matrix.npy", "confusion_matrix_normalized.npy",
    "top_confusions.json", "predictions.npy",
    "test_labels.npy", "test_track_ids.npy",
    "run_manifest.json", "model.joblib",
    "grid_search_results.csv", "grid_search_results.json",
]

CLASSICAL_MODELS = {"svm_rbf", "random_forest", "xgboost"}
DEEP_MODELS = {"bilstm", "cnn2d", "transformer", "mamba1", "mamba2"}
MAMBA_MODELS = {"mamba1", "mamba2"}

EXPECTED_TEST_SIZE = 1772


# ----- inventory --------------------------------------------------------
def discover_runs() -> dict[tuple[str, int], dict[str, Any]]:
    """Map (model_key, seed) -> dict(path, kind, provider)."""
    runs: dict[tuple[str, int], dict[str, Any]] = {}

    if PROVIDER_EXPORTS.is_dir():
        for prov_dir in PROVIDER_EXPORTS.iterdir():
            if not prov_dir.is_dir():
                continue
            provider = prov_dir.name
            for seed_dir in prov_dir.glob("results/final/*/*/*"):
                if not seed_dir.is_dir():
                    continue
                m = re.match(r"^([a-z0-9]+)_seed(\d+)$", seed_dir.name)
                if not m:
                    continue
                model = m.group(1)
                seed = int(m.group(2))
                key = (model, seed)
                if key in runs:
                    # duplicate; prefer the one with fresher mtime
                    existing = runs[key]["path"]
                    if seed_dir.stat().st_mtime > existing.stat().st_mtime:
                        runs[key] = {"path": seed_dir, "kind": "deep", "provider": provider}
                else:
                    runs[key] = {"path": seed_dir, "kind": "deep", "provider": provider}

    # classical baselines
    for child in FINAL_DIR.iterdir():
        m = re.match(r"^(svm|random_forest|xgboost)_seed(\d+)$", child.name)
        if not m or not child.is_dir():
            continue
        raw_model = m.group(1)
        # canonicalize: "svm" dir -> "svm_rbf" model_key (matches run_manifest.json)
        model_key = "svm_rbf" if raw_model == "svm" else raw_model
        seed = int(m.group(2))
        runs[(model_key, seed)] = {"path": child, "kind": "classical", "provider": "local"}

    return runs


# ----- helpers ----------------------------------------------------------
def file_ok(p: Path) -> tuple[bool, str]:
    if not p.exists():
        return False, "missing"
    if not p.is_file():
        return False, "not_a_file"
    if p.stat().st_size == 0:
        return False, "empty"
    return True, ""


def sha256_short(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def array_has_bad(arr: np.ndarray) -> tuple[bool, int]:
    if arr.dtype.kind in ("i", "u", "b"):
        return False, 0
    bad = ~np.isfinite(arr)
    return bool(bad.any()), int(bad.sum())


def parse_train_log(log_path: Path) -> dict[str, Any]:
    """Parse train.log for best_epoch, last epoch, final train_loss is nan, total epochs."""
    txt = log_path.read_text(errors="ignore")
    epochs = re.findall(r"epoch=(\d+)/(\d+)", txt)
    stop_epoch = int(epochs[-1][0]) if epochs else 0
    total_epochs = int(epochs[-1][1]) if epochs else 0
    bests = re.findall(r"best_val_macro_f1=([\d.]+|nan)\s+best_epoch=(\d+)\s+patience=(\d+)/(\d+)",
                       txt)
    best_epoch = int(bests[-1][1]) if bests else 0
    best_val = float(bests[-1][0]) if bests and bests[-1][0] != "nan" else float("nan")
    final_patience = int(bests[-1][2]) if bests else 0
    max_patience = int(bests[-1][3]) if bests else 0
    last_train_loss_nan = bool(re.search(r"epoch=\d+/\d+\s*\ntrain loss=nan",
                                         txt.split("epoch=")[-1] if "epoch=" in txt
                                         else txt))
    # cleaner: get final train loss line
    final_train_loss_match = re.findall(r"train loss=([\w.-]+)", txt)
    final_train_loss = final_train_loss_match[-1] if final_train_loss_match else ""
    last_train_loss_nan = final_train_loss == "nan"
    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "stop_epoch": stop_epoch,
        "total_epochs": total_epochs,
        "final_patience": final_patience,
        "max_patience": max_patience,
        "last_train_loss_nan": last_train_loss_nan,
        "patience_exhausted": final_patience >= max_patience and max_patience > 0,
    }


def torch_load_check(pt_path: Path) -> tuple[bool, int, str]:
    """Load checkpoint on CPU and return (ok, nan_tensor_count, err)."""
    try:
        import torch
        try:
            obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        return False, -1, f"torch.load failed: {e!r}"
    state = obj
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    nan_tensors = 0
    if isinstance(state, dict):
        import torch
        for _, v in state.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                if not torch.isfinite(v).all():
                    nan_tensors += 1
    return True, nan_tensors, ""


# ----- validation -------------------------------------------------------
def validate_deep(run_dir: Path, model: str, seed: int) -> dict[str, Any]:
    res: dict[str, Any] = {"failures": [], "info": {}}
    # files
    for name in DEEP_REQUIRED:
        ok, why = file_ok(run_dir / name)
        if not ok:
            res["failures"].append(f"{name}:{why}")
    if res["failures"]:
        return res

    # arrays
    preds = np.load(run_dir / "predictions.npy")
    probs = np.load(run_dir / "probabilities.npy")
    track_ids = np.load(run_dir / "test_track_ids.npy")
    labels = np.load(run_dir / "test_labels.npy")

    p_bad, p_n = array_has_bad(preds)
    q_bad, q_n = array_has_bad(probs)
    res["info"]["predictions_nan_count"] = p_n
    res["info"]["probabilities_nan_count"] = q_n
    if p_bad:
        res["failures"].append(
            f"predictions.npy contains {p_n} NaN/Inf values out of {preds.size}")
    if q_bad:
        res["failures"].append(
            f"probabilities.npy contains {q_n} NaN/Inf values out of {probs.size}")

    if preds.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"predictions.npy rows={preds.shape[0]} != {EXPECTED_TEST_SIZE}")
    if track_ids.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"test_track_ids.npy rows={track_ids.shape[0]} != {EXPECTED_TEST_SIZE}")
    if labels.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"test_labels.npy rows={labels.shape[0]} != {EXPECTED_TEST_SIZE}")

    # checkpoint load
    ok, nan_tensors, err = torch_load_check(run_dir / "best.pt")
    res["info"]["best_pt_loads"] = ok
    res["info"]["best_pt_nan_tensors"] = nan_tensors
    if not ok:
        res["failures"].append(f"best.pt load failed: {err}")
    elif nan_tensors > 0:
        res["failures"].append(f"best.pt contains {nan_tensors} non-finite tensors")

    # train log
    log_info = parse_train_log(run_dir / "train.log")
    res["info"].update(log_info)

    # metrics.json sanity
    metrics = json.loads((run_dir / "metrics.json").read_text())
    test_macro_f1 = metrics.get("test_macro_f1")
    best_val = log_info["best_val_macro_f1"]
    res["info"]["test_macro_f1"] = test_macro_f1
    res["info"]["best_val_macro_f1_metrics"] = metrics.get("best_val_macro_f1")

    # mamba-specific: best_epoch must precede nan tail; checkpoint is best.pt
    if model in MAMBA_MODELS:
        if log_info["stop_epoch"] < log_info["best_epoch"]:
            res["failures"].append("stop_epoch precedes best_epoch (impossible)")
        # require patience exhaustion (or best at the very last epoch w/o NaN)
        if log_info["last_train_loss_nan"] and not log_info["patience_exhausted"]:
            res["failures"].append(
                "training ended on NaN loss without patience exhaustion")
        # plausibility: test_macro_f1 >= 0.7 * best_val_macro_f1 (if best_val finite)
        if (best_val == best_val and test_macro_f1 is not None  # not nan
                and test_macro_f1 < 0.7 * best_val):
            res["failures"].append(
                f"test_macro_f1={test_macro_f1:.4f} < 0.7 * best_val_macro_f1={best_val:.4f}")

    return res


def validate_classical(run_dir: Path, model: str, seed: int) -> dict[str, Any]:
    res: dict[str, Any] = {"failures": [], "info": {}}
    required = list(CLASSICAL_REQUIRED_BASE)
    if model != "svm_rbf":
        required.append("probabilities.npy")
    else:
        required.append("decision_scores.npy")
    for name in required:
        ok, why = file_ok(run_dir / name)
        if not ok:
            res["failures"].append(f"{name}:{why}")
    if res["failures"]:
        return res

    preds = np.load(run_dir / "predictions.npy")
    track_ids = np.load(run_dir / "test_track_ids.npy")
    labels = np.load(run_dir / "test_labels.npy")
    p_bad, p_n = array_has_bad(preds)
    res["info"]["predictions_nan_count"] = p_n
    if p_bad:
        res["failures"].append(
            f"predictions.npy contains {p_n} NaN/Inf values out of {preds.size}")
    if preds.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"predictions.npy rows={preds.shape[0]} != {EXPECTED_TEST_SIZE}")
    if track_ids.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"test_track_ids.npy rows={track_ids.shape[0]} != {EXPECTED_TEST_SIZE}")
    if labels.shape[0] != EXPECTED_TEST_SIZE:
        res["failures"].append(
            f"test_labels.npy rows={labels.shape[0]} != {EXPECTED_TEST_SIZE}")

    if model == "svm_rbf":
        scores = np.load(run_dir / "decision_scores.npy")
        s_bad, s_n = array_has_bad(scores)
        if s_bad:
            res["failures"].append(
                f"decision_scores.npy contains {s_n} NaN/Inf values")
    else:
        probs = np.load(run_dir / "probabilities.npy")
        q_bad, q_n = array_has_bad(probs)
        res["info"]["probabilities_nan_count"] = q_n
        if q_bad:
            res["failures"].append(
                f"probabilities.npy contains {q_n} NaN/Inf values out of {probs.size}")

    metrics = json.loads((run_dir / "metrics.json").read_text())
    res["info"]["test_macro_f1"] = (metrics.get("macro_f1")
                                    or metrics.get("test_metrics", {}).get("macro_f1")
                                    or metrics.get("test_macro_f1"))
    return res


# ----- main -------------------------------------------------------------
def main() -> int:
    if not SPLIT_MANIFEST.is_file():
        print(f"FATAL: split manifest missing: {SPLIT_MANIFEST}")
        return 1

    # confirm canonical test size
    with SPLIT_MANIFEST.open(newline="") as f:
        rdr = csv.DictReader(f)
        canonical_test = sum(1 for r in rdr if r.get("split") == "test")
    if canonical_test != EXPECTED_TEST_SIZE:
        print(f"WARNING: canonical test split size {canonical_test} != {EXPECTED_TEST_SIZE}")

    runs = discover_runs()
    print(f"Inventoried {len(runs)} candidate (model, seed) runs")
    deep_n = sum(1 for v in runs.values() if v["kind"] == "deep")
    classical_n = sum(1 for v in runs.values() if v["kind"] == "classical")
    print(f"  deep models:      {deep_n}")
    print(f"  classical models: {classical_n}")

    admitted: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    mamba_audit: list[dict[str, Any]] = []

    for (model, seed), info in sorted(runs.items()):
        run_dir: Path = info["path"]
        kind = info["kind"]
        provider = info["provider"]
        try:
            if kind == "deep":
                res = validate_deep(run_dir, model, seed)
            else:
                res = validate_classical(run_dir, model, seed)
        except Exception as e:
            res = {"failures": [f"validator_exception: {e!r}\n{traceback.format_exc()}"],
                   "info": {}}

        # mamba audit row
        if model in MAMBA_MODELS:
            li = res["info"]
            row = {
                "model": model, "seed": seed,
                "best_epoch": li.get("best_epoch"),
                "stop_epoch": li.get("stop_epoch"),
                "last_train_loss_was_nan": li.get("last_train_loss_nan"),
                "patience_exhausted": li.get("patience_exhausted"),
                "predictions_npy_has_nan": li.get("predictions_nan_count", 0) > 0,
                "probabilities_npy_has_nan": li.get("probabilities_nan_count", 0) > 0,
                "best_pt_loads": li.get("best_pt_loads"),
                "best_pt_nan_tensors": li.get("best_pt_nan_tensors"),
                "test_macro_f1": li.get("test_macro_f1"),
                "best_val_macro_f1": li.get("best_val_macro_f1"),
                "admissible": not res["failures"],
            }
            mamba_audit.append(row)

        rel = run_dir.relative_to(REPO_ROOT).as_posix()
        if not res["failures"]:
            # gather extra fields
            envj = json.loads((run_dir / "environment.json").read_text())
            mfj = json.loads((run_dir / "run_manifest.json").read_text())
            metricsj = json.loads((run_dir / "metrics.json").read_text())
            gpu = envj.get("gpu_name") or mfj.get("gpu") or ""
            elapsed = (metricsj.get("total_train_seconds")
                       or mfj.get("fit_seconds")
                       or metricsj.get("test_metrics", {}).get("fit_seconds"))
            cmd = mfj.get("command") or mfj.get("command_template") or ""
            best_pt = run_dir / ("best.pt" if kind == "deep" else "model.joblib")
            sha = sha256_short(best_pt) if best_pt.is_file() else ""
            entry = {
                "model_key": model,
                "seed": seed,
                "kind": kind,
                "provider": provider,
                "output_dir_abs": str(run_dir),
                "output_dir_repo_relative": rel,
                "gpu": gpu,
                "wall_time_seconds": elapsed,
                "command": cmd,
                "source_path": str(run_dir),
                "copied_back_path": str(run_dir),
                "best_pt_sha256_16": sha,
                "test_macro_f1": (metricsj.get("test_macro_f1")
                                  or metricsj.get("macro_f1")
                                  or metricsj.get("test_metrics", {}).get("macro_f1")),
                "validation_passed": True,
            }
            admitted.append(entry)
        else:
            reason = "; ".join(res["failures"])
            skip_entry = {
                "model_key": model,
                "seed": seed,
                "kind": kind,
                "provider": provider,
                "output_dir_abs": str(run_dir),
                "output_dir_repo_relative": rel,
                "reason": reason,
                "info": res["info"],
            }
            if model in MAMBA_MODELS:
                skip_entry["refit_ticket"] = f"mamba_stable_refit__{model}__seed{seed}"
            skips.append(skip_entry)

    # write outputs
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUT_MANIFEST.write_text(json.dumps({
        "schema_version": 1,
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "canonical_test_size": EXPECTED_TEST_SIZE,
        "admitted_count": len(admitted),
        "entries": admitted,
    }, indent=2))
    OUT_SKIPS.write_text(json.dumps({
        "schema_version": 1,
        "generated_at_utc": dt.datetime.utcnow().isoformat() + "Z",
        "skipped_count": len(skips),
        "entries": skips,
    }, indent=2))

    # mamba audit table
    print("\n=== MAMBA NaN AUDIT ===")
    hdr = ("model", "seed", "best_ep", "stop_ep", "train_nan", "pat_exh",
           "pred_nan", "prob_nan", "best_pt_ok", "test_f1", "best_val_f1", "admit")
    print("{:<7} {:<4} {:<7} {:<7} {:<9} {:<7} {:<8} {:<8} {:<10} {:<7} {:<10} {:<5}".format(*hdr))
    for r in sorted(mamba_audit, key=lambda x: (x["model"], x["seed"])):
        print("{:<7} {:<4} {:<7} {:<7} {:<9} {:<7} {:<8} {:<8} {:<10} {:<7} {:<10} {:<5}".format(
            r["model"], r["seed"],
            str(r["best_epoch"]), str(r["stop_epoch"]),
            str(r["last_train_loss_was_nan"]), str(r["patience_exhausted"]),
            str(r["predictions_npy_has_nan"]), str(r["probabilities_npy_has_nan"]),
            str(r["best_pt_loads"]),
            f"{r['test_macro_f1']:.4f}" if isinstance(r["test_macro_f1"], (int, float)) else "n/a",
            f"{r['best_val_macro_f1']:.4f}" if isinstance(r["best_val_macro_f1"], (int, float)) and r["best_val_macro_f1"] == r["best_val_macro_f1"] else "n/a",
            "Y" if r["admissible"] else "N",
        ))

    # per-model means (admissible)
    by_model: dict[str, list[float]] = {}
    for e in admitted:
        v = e.get("test_macro_f1")
        if isinstance(v, (int, float)):
            by_model.setdefault(e["model_key"], []).append(float(v))
    print("\n=== PER-MODEL MACRO F1 MEAN (admissible seeds) ===")
    for m in sorted(by_model):
        vs = by_model[m]
        print(f"  {m:<14} n={len(vs)}  mean={sum(vs)/len(vs):.4f}  "
              f"min={min(vs):.4f}  max={max(vs):.4f}")

    # audit log
    sha = hashlib.sha256(OUT_MANIFEST.read_bytes()).hexdigest()
    with OUT_AUDIT.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 72 + "\n")
        f.write(f"frozen-manifest-audit @ {dt.datetime.utcnow().isoformat()}Z\n")
        f.write("=" * 72 + "\n")
        f.write(f"candidates_scanned: {len(runs)}\n")
        f.write(f"  deep: {deep_n}  classical: {classical_n}\n")
        f.write(f"admitted: {len(admitted)}\n")
        f.write(f"skipped:  {len(skips)}\n")
        if skips:
            f.write("skip reasons:\n")
            for s in skips:
                f.write(f"  - {s['model_key']} seed{s['seed']}: {s['reason']}\n")
        f.write("per-model macro F1 mean (admissible):\n")
        for m in sorted(by_model):
            vs = by_model[m]
            f.write(f"  {m:<14} n={len(vs)}  mean={sum(vs)/len(vs):.4f}\n")
        f.write(f"manifest_path: {OUT_MANIFEST}\n")
        f.write(f"manifest_sha256: {sha}\n")
        f.write(f"skips_path: {OUT_SKIPS}\n")

    print(f"\nWrote {OUT_MANIFEST}  ({len(admitted)} entries)")
    print(f"Wrote {OUT_SKIPS}  ({len(skips)} entries)")
    print(f"Audit log appended to {OUT_AUDIT}")
    print(f"manifest sha256: {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
