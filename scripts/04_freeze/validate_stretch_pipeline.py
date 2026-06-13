from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
VALIDATION_DIR = RESULTS_DIR / "validation"
REPORT_PATH = VALIDATION_DIR / "stretch_pipeline_validation.json"

EXPECTED_TEST_TRACKS = 1772
EXPECTED_VAL_TRACKS = 1704
EXPECTED_TRAIN_TRACKS = 13512
EXPECTED_MEL_SHAPES = {
    60: (128, 2584),
    120: (128, 5168),
}

DEEP_CORE_ARTIFACTS = (
    "config.json",
    "environment.json",
    "debug_gates.json",
    "history.csv",
    "history.jsonl",
    "train.log",
    "metrics.json",
    "classification_report.json",
    "report.txt",
    "confusion_matrix.npy",
    "confusion_matrix_normalized.npy",
    "top_confusions.json",
    "predictions.npy",
    "probabilities.npy",
    "test_labels.npy",
    "test_track_ids.npy",
    "run_manifest.json",
)

MODEL_OUTPUT_NAME = {"lstm": "bilstm"}
LEARNING_MODELS = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")
LEARNING_FRACTIONS = (25, 50, 75, 100)
ABLATION_MODELS = ("transformer", "mamba2")
ABLATION_SEEDS = (42, 43, 44)


@dataclass
class Check:
    name: str
    status: str
    detail: str
    path: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def finite_metric(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def ok(name: str, detail: str, path: Path | None = None) -> Check:
    return Check(name=name, status="ok", detail=detail, path=rel(path) if path else None)


def fail(name: str, detail: str, path: Path | None = None) -> Check:
    return Check(name=name, status="fail", detail=detail, path=rel(path) if path else None)


def warn(name: str, detail: str, path: Path | None = None) -> Check:
    return Check(name=name, status="warn", detail=detail, path=rel(path) if path else None)


def validate_file(path: Path, name: str) -> Check:
    if not path.exists():
        return fail(name, "missing", path)
    if path.is_file() and path.stat().st_size <= 0:
        return fail(name, "empty file", path)
    return ok(name, "present", path)


def validate_arrays(run_dir: Path, expected_test_tracks: int) -> list[Check]:
    checks: list[Check] = []
    arrays: dict[str, np.ndarray] = {}
    for name in ("predictions.npy", "probabilities.npy", "test_labels.npy", "test_track_ids.npy"):
        path = run_dir / name
        if not path.exists():
            checks.append(fail(f"{rel(run_dir)}:{name}", "missing", path))
            continue
        try:
            arr = np.load(path, allow_pickle=False)
        except Exception as exc:
            checks.append(fail(f"{rel(run_dir)}:{name}", f"could not load array: {exc}", path))
            continue
        arrays[name] = arr
        if not np.isfinite(arr).all():
            checks.append(fail(f"{rel(run_dir)}:{name}", "contains non-finite values", path))
        else:
            checks.append(ok(f"{rel(run_dir)}:{name}", f"shape={list(arr.shape)}", path))
    for name in ("predictions.npy", "test_labels.npy", "test_track_ids.npy"):
        arr = arrays.get(name)
        if arr is not None and int(arr.shape[0]) != expected_test_tracks:
            checks.append(fail(f"{rel(run_dir)}:{name}:row_count", f"expected {expected_test_tracks}, found {arr.shape[0]}", run_dir / name))
    probs = arrays.get("probabilities.npy")
    if probs is not None:
        if probs.ndim != 2 or probs.shape[0] != expected_test_tracks:
            checks.append(fail(f"{rel(run_dir)}:probabilities.npy:shape", f"expected first dim {expected_test_tracks}, found {list(probs.shape)}", run_dir / "probabilities.npy"))
        elif probs.shape[1] != 16:
            checks.append(fail(f"{rel(run_dir)}:probabilities.npy:classes", f"expected 16 classes, found {probs.shape[1]}", run_dir / "probabilities.npy"))
    return checks


def validate_run_dir(
    run_dir: Path,
    *,
    expected_model: str | None = None,
    expected_seed: int | None = None,
    expected_crop_seconds: int | None = None,
    require_checkpoint: bool = False,
    expected_test_tracks: int = EXPECTED_TEST_TRACKS,
    load_arrays: bool = True,
) -> list[Check]:
    checks: list[Check] = []
    if not run_dir.exists():
        return [fail(f"{rel(run_dir)}:run_dir", "missing run directory", run_dir)]
    for artifact in DEEP_CORE_ARTIFACTS:
        checks.append(validate_file(run_dir / artifact, f"{rel(run_dir)}:{artifact}"))
    checkpoint = run_dir / "best.pt"
    if require_checkpoint:
        checks.append(validate_file(checkpoint, f"{rel(run_dir)}:best.pt"))
    elif not checkpoint.exists():
        # published bundles ship artifacts without the heavy checkpoint, so we warn rather
        # than fail unless the caller explicitly asked us to require it.
        checks.append(warn(f"{rel(run_dir)}:best.pt", "checkpoint absent; acceptable only for exported paper bundles", checkpoint))
    manifest = load_json(run_dir / "run_manifest.json")
    metrics = load_json(run_dir / "metrics.json")
    if not isinstance(manifest, dict):
        checks.append(fail(f"{rel(run_dir)}:manifest_json", "run_manifest.json is not an object", run_dir / "run_manifest.json"))
    else:
        status = str(manifest.get("status") or "")
        if status not in {"completed", "complete"}:
            checks.append(fail(f"{rel(run_dir)}:status", f"expected completed, found {status!r}", run_dir / "run_manifest.json"))
        else:
            checks.append(ok(f"{rel(run_dir)}:status", status, run_dir / "run_manifest.json"))
        # the trainer records bilstm runs under the legacy "lstm" key, so we accept either
        # spelling when checking the recorded model name.
        if expected_model and str(manifest.get("model")) not in {expected_model, "lstm" if expected_model == "bilstm" else expected_model}:
            checks.append(fail(f"{rel(run_dir)}:model", f"expected {expected_model}, found {manifest.get('model')}", run_dir / "run_manifest.json"))
        try:
            seed_value = int(manifest.get("seed", -1))
        except (TypeError, ValueError):
            seed_value = -1
        if expected_seed is not None and seed_value != expected_seed:
            checks.append(fail(f"{rel(run_dir)}:seed", f"expected {expected_seed}, found {manifest.get('seed')}", run_dir / "run_manifest.json"))
        if expected_crop_seconds is not None:
            try:
                crop = int(float(manifest.get("crop_seconds")))
            except (TypeError, ValueError):
                crop = -1
            if crop != expected_crop_seconds:
                checks.append(fail(f"{rel(run_dir)}:crop_seconds", f"expected {expected_crop_seconds}, found {manifest.get('crop_seconds')}", run_dir / "run_manifest.json"))
    if not isinstance(metrics, dict):
        checks.append(fail(f"{rel(run_dir)}:metrics_json", "metrics.json is not an object", run_dir / "metrics.json"))
    else:
        macro = finite_metric(metrics, "test_macro_f1", "macro_f1")
        acc = finite_metric(metrics, "test_accuracy", "accuracy")
        if macro is None:
            checks.append(fail(f"{rel(run_dir)}:macro_f1", "missing or non-finite", run_dir / "metrics.json"))
        else:
            checks.append(ok(f"{rel(run_dir)}:macro_f1", f"{macro:.6f}", run_dir / "metrics.json"))
        if acc is None:
            checks.append(fail(f"{rel(run_dir)}:accuracy", "missing or non-finite", run_dir / "metrics.json"))
    if load_arrays:
        checks.extend(validate_arrays(run_dir, expected_test_tracks))
    return checks


def expected_ablation_dirs() -> list[tuple[str, int, int, Path]]:
    return [
        (model, seed, 15, RESULTS_DIR / "ablation" / f"{model}_seed{seed}_15s")
        for model in ABLATION_MODELS
        for seed in ABLATION_SEEDS
    ]


def expected_learning_dirs() -> list[tuple[str, int, Path]]:
    return [
        (model, fraction, RESULTS_DIR / "learning_curves" / f"{model}_seed42_frac{fraction}")
        for model in LEARNING_MODELS
        for fraction in LEARNING_FRACTIONS
    ]


def expected_long_sequence_dirs(profile: str, include_transformer_120: bool) -> list[tuple[str, int, int, Path]]:
    specs: list[tuple[str, int, int]] = [("mamba2", 42, 60)]
    if profile in {"overnight-full", "extended"}:
        specs.extend([("transformer", 42, 60), ("mamba2", 42, 120)])
    if profile == "extended":
        specs.extend(("mamba2", seed, 60) for seed in (43, 44, 45, 46))
        specs.extend(("transformer", seed, 60) for seed in (43, 44))
        specs.extend(("mamba2", seed, 120) for seed in (43, 44))
    if include_transformer_120:
        specs.append(("transformer", 42, 120))
    return [
        (model, seed, seconds, RESULTS_DIR / "long_sequence" / f"{model}_seed{seed}_{seconds}s")
        for model, seed, seconds in specs
    ]


def validate_stretch_runs(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    for model, seed, seconds, run_dir in expected_ablation_dirs():
        checks.extend(validate_run_dir(run_dir, expected_model=model, expected_seed=seed, expected_crop_seconds=seconds, require_checkpoint=args.require_checkpoints, load_arrays=not args.fast))
    for model, fraction, run_dir in expected_learning_dirs():
        checks.extend(validate_run_dir(run_dir, expected_model=model, expected_seed=42, expected_crop_seconds=30, require_checkpoint=args.require_checkpoints, load_arrays=not args.fast))
    for model, seed, seconds, run_dir in expected_long_sequence_dirs(args.profile, args.include_transformer_120):
        checks.extend(validate_run_dir(run_dir, expected_model=model, expected_seed=seed, expected_crop_seconds=seconds, require_checkpoint=args.require_checkpoints, load_arrays=not args.fast))
    return checks


def validate_manifests(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    frozen = load_json(RESULTS_DIR / "frozen_manifest.json")
    if not isinstance(frozen, dict):
        checks.append(fail("frozen_manifest", "missing or invalid", RESULTS_DIR / "frozen_manifest.json"))
    else:
        admitted = int(frozen.get("admitted_count") or len(frozen.get("entries", [])))
        checks.append(ok("frozen_manifest", f"admitted_count={admitted}", RESULTS_DIR / "frozen_manifest.json") if admitted >= 40 else fail("frozen_manifest", f"expected >=40 admitted, found {admitted}", RESULTS_DIR / "frozen_manifest.json"))

    ablation = load_json(RESULTS_DIR / "ablation_manifest.json")
    if isinstance(ablation, dict):
        completed = int(ablation.get("completed_count") or 0)
        checks.append(ok("ablation_manifest", f"completed_count={completed}", RESULTS_DIR / "ablation_manifest.json") if completed >= 6 else fail("ablation_manifest", f"expected >=6 completed, found {completed}", RESULTS_DIR / "ablation_manifest.json"))
    else:
        checks.append(fail("ablation_manifest", "missing", RESULTS_DIR / "ablation_manifest.json"))

    learning = load_json(RESULTS_DIR / "learning_curves" / "aggregated.json")
    if isinstance(learning, dict):
        count = sum(len(points) for points in learning.values() if isinstance(points, dict))
        checks.append(ok("learning_curves_aggregated", f"entries={count}", RESULTS_DIR / "learning_curves" / "aggregated.json") if count >= 20 else fail("learning_curves_aggregated", f"expected 20 entries, found {count}", RESULTS_DIR / "learning_curves" / "aggregated.json"))
    else:
        checks.append(fail("learning_curves_aggregated", "missing", RESULTS_DIR / "learning_curves" / "aggregated.json"))

    long_manifest = load_json(RESULTS_DIR / "long_sequence" / "manifest.json")
    if isinstance(long_manifest, dict):
        completed = int(long_manifest.get("completed_count") or 0)
        minimum = len(expected_long_sequence_dirs(args.profile, args.include_transformer_120))
        checks.append(ok("long_sequence_manifest", f"completed_count={completed}", RESULTS_DIR / "long_sequence" / "manifest.json") if completed >= minimum else fail("long_sequence_manifest", f"expected >={minimum} completed, found {completed}", RESULTS_DIR / "long_sequence" / "manifest.json"))
    else:
        checks.append(fail("long_sequence_manifest", "missing", RESULTS_DIR / "long_sequence" / "manifest.json"))
    return checks


def validate_paper_assets(args: argparse.Namespace) -> list[Check]:
    checks: list[Check] = []
    required = [
        "table1_performance.csv",
        "table2_efficiency_calibration.csv",
        "table3_top_confusions.csv",
        "manifest.json",
        "figures/fig1_inter_arch_cka.png",
        "figures/fig2_8group_cka_profile.png",
        "figures/fig3_cka_vs_f1.png",
        "figures/fig4_novelty_gap.png",
        "table4_ablation.csv",
        "figures/fig6_learning_curves.png",
        "table5_long_sequence.csv",
    ]
    for item in required:
        checks.append(validate_file(RESULTS_DIR / "paper_assets" / item, f"paper_assets:{item}"))
    return checks


def validate_fma_full_data() -> list[Check]:
    checks: list[Check] = []
    raw_root = REPO_ROOT / "data" / "fma_full" / "raw"
    if raw_root.exists():
        mp3_count = len(list(raw_root.rglob("*.mp3")))
        checks.append(ok("fma_full_raw_mp3_count", f"mp3_count={mp3_count}", raw_root) if mp3_count >= EXPECTED_TEST_TRACKS + EXPECTED_VAL_TRACKS else warn("fma_full_raw_mp3_count", f"partial or not local: mp3_count={mp3_count}", raw_root))
    else:
        checks.append(warn("fma_full_raw", "not present locally; remote/provider validation required", raw_root))
    for seconds, expected_shape in EXPECTED_MEL_SHAPES.items():
        root = REPO_ROOT / "data" / "processed" / f"fma_full_{seconds}s"
        for split, expected_count in (("test", EXPECTED_TEST_TRACKS), ("validation", EXPECTED_VAL_TRACKS), ("training", EXPECTED_TRAIN_TRACKS)):
            split_dir = root / split
            if not split_dir.exists():
                checks.append(warn(f"fma_full_{seconds}s_{split}", "not present locally", split_dir))
                continue
            paths = list(split_dir.glob("*.npy"))
            status_fn = ok if len(paths) >= expected_count else warn
            checks.append(status_fn(f"fma_full_{seconds}s_{split}_count", f"npy_count={len(paths)} expected={expected_count}", split_dir))
            if paths:
                try:
                    shape = tuple(np.load(paths[0], mmap_mode="r").shape)
                except Exception as exc:
                    checks.append(fail(f"fma_full_{seconds}s_{split}_shape", f"could not read sample: {exc}", paths[0]))
                else:
                    checks.append(ok(f"fma_full_{seconds}s_{split}_shape", f"shape={shape}", paths[0]) if shape == expected_shape else fail(f"fma_full_{seconds}s_{split}_shape", f"expected {expected_shape}, found {shape}", paths[0]))
    return checks


def summarize(checks: list[Check]) -> dict[str, Any]:
    counts = {"ok": 0, "warn": 0, "fail": 0}
    for check in checks:
        counts[check.status] = counts.get(check.status, 0) + 1
    return {
        "created_at_utc": utc_now(),
        "status": "failed" if counts.get("fail", 0) else "passed_with_warnings" if counts.get("warn", 0) else "passed",
        "counts": counts,
        "checks": [check.__dict__ for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate stretch pipeline artifacts and paper-asset gates.")
    parser.add_argument("--profile", choices=["paper-minimum", "overnight-full", "extended"], default="overnight-full")
    parser.add_argument("--include-transformer-120", action="store_true")
    parser.add_argument("--require-checkpoints", action="store_true")
    parser.add_argument("--fast", action="store_true", help="skip numpy array loads")
    parser.add_argument("--no-fail", action="store_true", help="write report and return zero even when checks fail")
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    args = parser.parse_args()

    checks: list[Check] = []
    checks.extend(validate_manifests(args))
    checks.extend(validate_stretch_runs(args))
    checks.extend(validate_paper_assets(args))
    checks.extend(validate_fma_full_data())
    report = summarize(checks)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": rel(args.report), "status": report["status"], "counts": report["counts"]}, indent=2, sort_keys=True))
    if report["counts"].get("fail", 0) and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
