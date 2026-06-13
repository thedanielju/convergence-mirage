from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.classical import (
    DEFAULT_REFIT_SCOPE,
    evaluate_svm,
    final_recommendation,
    run_grid_search,
    set_classical_seed,
    write_classical_run_manifest,
)
from src.training.logging import collect_environment, prepare_output_dir, to_jsonable, write_json


FULL_GRID = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto", 1e-3, 1e-2],
}
SMOKE_GRID = {
    "svm__C": [1],
    "svm__gamma": ["scale"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"run_{timestamp}_svm"


def main() -> None:
    parser = argparse.ArgumentParser(description="SVM baseline for FMA medium classical features")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="run a tiny grid for artifact verification")
    parser.add_argument("--n-jobs", type=int, default=1, help="n_jobs passed to GridSearchCV")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--refit-scope", choices=["train_only", "train_validation"], default=DEFAULT_REFIT_SCOPE)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing non-empty output directory")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else default_output_dir()
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    set_classical_seed(args.seed)
    grid = SMOKE_GRID if args.smoke else FULL_GRID

    write_json(
        output_dir / "config.json",
        {
            "model": "svm_rbf",
            "seed": args.seed,
            "feature_matrix": "540-d standardized FMA+rhythm features",
            "selection_metric": "validation_macro_f1",
            "refit_scope": args.refit_scope,
            "refit_scope_recommendation": final_recommendation(args.refit_scope),
            "kernel": "rbf",
            "class_weight": "balanced",
            "grid": grid,
            "probability": False,
            "smoke": args.smoke,
            "n_jobs": args.n_jobs,
        },
    )
    write_json(output_dir / "environment.json", collect_environment())
    write_classical_run_manifest(output_dir, {"status": "running", "model": "svm_rbf", "seed": args.seed, "started_at_utc": utc_now()})

    artifacts = run_grid_search(
        timeout_seconds=args.timeout_seconds,
        output_dir=output_dir,
        param_grid=grid,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
        seed=args.seed,
        refit_scope=args.refit_scope,
    )
    metrics = evaluate_svm(artifacts.model, output_dir)
    payload = {
        "output_dir": str(output_dir),
        "used_small_grid": artifacts.used_small_grid,
        "best_params": artifacts.best_params,
        "validation_macro_f1": artifacts.best_score,
        "fit_seconds": artifacts.fit_seconds,
        "refit_scope": artifacts.refit_scope,
        "test_metrics": metrics,
    }
    write_json(output_dir / "run_summary.json", payload)
    write_classical_run_manifest(output_dir, {"status": "completed", "model": "svm_rbf", "seed": args.seed, "finished_at_utc": utc_now(), **payload})
    print(json.dumps(to_jsonable(payload), indent=2))


if __name__ == "__main__":
    main()
