from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cka import (
    align_matrices_by_track_id,
    correlation_rows,
    linear_cka,
    load_split_features,
    matrix_table,
    pairwise_cka_table,
    rbf_cka,
    resolve_feature_group_indices,
    run_smoke_checks,
    save_csv,
    save_matrix_csv,
    validate_matrix,
    write_feature_index_manifest,
)
from src.analysis.representations import extract_untrained_representations, load_representation_dir


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def representation_name(rep: dict[str, Any], fallback: str) -> str:
    manifest = rep.get("manifest") or {}
    model = manifest.get("model") or fallback
    seed = manifest.get("seed")
    source_run = manifest.get("source_run_dir")
    if seed is None and source_run:
        metrics = read_json(Path(source_run) / "metrics.json")
        if metrics:
            seed = metrics.get("seed")
    suffix = f"seed{seed}" if seed is not None else Path(str(rep["path"])).name
    return f"{model}_{suffix}"


def metric_from_run(rep: dict[str, Any]) -> dict[str, float | None]:
    manifest = rep.get("manifest") or {}
    candidates: list[Path] = []
    if manifest.get("source_run_dir"):
        candidates.append(Path(str(manifest["source_run_dir"])))
    candidates.append(Path(rep["path"]).parent)
    for directory in candidates:
        payload = read_json(directory / "metrics.json")
        if not payload:
            continue
        return {
            "accuracy": _float_or_none(payload.get("test_accuracy", payload.get("accuracy"))),
            "macro_f1": _float_or_none(payload.get("test_macro_f1", payload.get("macro_f1"))),
            "balanced_accuracy": _float_or_none(payload.get("test_balanced_accuracy", payload.get("balanced_accuracy"))),
        }
    return {"accuracy": None, "macro_f1": None, "balanced_accuracy": None}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def load_classical_reference(run_dir: Path) -> dict[str, Any] | None:
    cka_ref_path = run_dir / "cka_reference.json"
    if not cka_ref_path.exists():
        return None
    payload = read_json(cka_ref_path) or {}
    matrix_path = payload.get("matrix_path") or payload.get("features_path")
    track_ids_path = payload.get("track_ids_path")
    if not matrix_path or not track_ids_path:
        return None
    matrix_file = (run_dir / matrix_path).resolve() if not Path(matrix_path).is_absolute() else Path(matrix_path)
    ids_file = (run_dir / track_ids_path).resolve() if not Path(track_ids_path).is_absolute() else Path(track_ids_path)
    if not matrix_file.exists() or not ids_file.exists():
        return None
    return {"matrix": np.load(matrix_file), "track_ids": np.load(ids_file).astype(np.int64), "manifest": payload}


def compute_group_profiles(
    representation_matrices: dict[str, np.ndarray],
    feature_matrix: np.ndarray,
    group_indices: dict[str, list[int]],
    rbf_sample_size: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    linear_rows: list[dict[str, Any]] = []
    rbf_rows: list[dict[str, Any]] = []
    for rep_name, rep_matrix in representation_matrices.items():
        full_linear = linear_cka(rep_matrix, feature_matrix)
        full_rbf = rbf_cka(rep_matrix, feature_matrix, sample_size=rbf_sample_size, seed=seed)
        for group_name, indices in group_indices.items():
            group_matrix = feature_matrix[:, indices]
            linear_rows.append(
                {
                    "representation": rep_name,
                    "feature_group": group_name,
                    "group_width": len(indices),
                    "linear_cka": float(linear_cka(rep_matrix, group_matrix)),
                    "full_feature_linear_cka": float(full_linear),
                }
            )
            rbf_rows.append(
                {
                    "representation": rep_name,
                    "feature_group": group_name,
                    "group_width": len(indices),
                    "rbf_cka": float(rbf_cka(rep_matrix, group_matrix, sample_size=rbf_sample_size, seed=seed)),
                    "full_feature_rbf_cka": float(full_rbf),
                    "rbf_sample_size": rbf_sample_size,
                }
            )
    return linear_rows, rbf_rows


def compute_novelty_gap(profile_rows: list[dict[str, Any]], metrics: dict[str, dict[str, float | None]]) -> list[dict[str, Any]]:
    by_rep: dict[str, list[dict[str, Any]]] = {}
    for row in profile_rows:
        by_rep.setdefault(str(row["representation"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for rep_name, rows_for_rep in by_rep.items():
        # the novelty gap is how far the representation sits from its closest classical
        # feature group, so a large gap means the model learned something hand features miss.
        group_values = [float(row["linear_cka"]) for row in rows_for_rep]
        max_group = float(np.max(group_values))
        mean_group = float(np.mean(group_values))
        full_value = float(rows_for_rep[0]["full_feature_linear_cka"])
        metric = metrics.get(rep_name, {})
        rows.append(
            {
                "representation": rep_name,
                "full_feature_linear_cka": full_value,
                "max_group_linear_cka": max_group,
                "mean_group_linear_cka": mean_group,
                "novelty_gap_vs_best_group": float(1.0 - max_group),
                "full_minus_best_group": float(full_value - max_group),
                "accuracy": metric.get("accuracy"),
                "macro_f1": metric.get("macro_f1"),
                "balanced_accuracy": metric.get("balanced_accuracy"),
            }
        )
    return rows


def within_architecture_rows(rep_names: list[str], rep_manifests: dict[str, dict[str, Any]], matrices: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    by_model: dict[str, list[str]] = {}
    for name in rep_names:
        model = str(rep_manifests.get(name, {}).get("model", name.split("_")[0]))
        by_model.setdefault(model, []).append(name)
    rows: list[dict[str, Any]] = []
    for model, names in by_model.items():
        if len(names) < 2:
            rows.append({"model": model, "n_representations": len(names), "status": "skipped_needs_at_least_two_seeds", "mean_linear_cka": None})
            continue
        values = []
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                value = linear_cka(matrices[left], matrices[right])
                values.append(value)
                rows.append({"model": model, "left": left, "right": right, "linear_cka": float(value), "status": "computed", "mean_linear_cka": None})
        rows.append({"model": model, "n_representations": len(names), "status": "summary", "mean_linear_cka": float(np.mean(values))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CKA representation analysis")
    parser.add_argument("--representation-dir", type=Path, action="append", default=[])
    parser.add_argument("--classical-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--array-name", default="penultimate")
    parser.add_argument("--rbf-sample-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--compute-untrained-floor", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        result = run_smoke_checks()
        print(json.dumps(result, indent=2))
        return

    if not args.representation_dir:
        raise ValueError("pass at least one --representation-dir, or use --smoke-test")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (REPO_ROOT / "results" / "analysis" / f"cka_{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix, feature_labels, feature_track_ids = load_split_features(args.split)
    group_indices = resolve_feature_group_indices()
    write_feature_index_manifest(output_dir / "feature_group_index_manifest.json")

    representation_payloads = [load_representation_dir(path, array_name=args.array_name) for path in args.representation_dir]
    matrices_for_alignment: dict[str, tuple[np.ndarray, np.ndarray]] = {"classical_full": (feature_track_ids, feature_matrix)}
    rep_manifests: dict[str, dict[str, Any]] = {}
    rep_paths: dict[str, Path] = {}
    for index, rep in enumerate(representation_payloads, start=1):
        name = representation_name(rep, fallback=f"representation{index}")
        if name in matrices_for_alignment:
            name = f"{name}_{index}"
        matrices_for_alignment[name] = (rep["track_ids"], rep["matrix"])
        rep_manifests[name] = rep.get("manifest") or {}
        rep_paths[name] = Path(rep["path"])

    for run_dir in args.classical_run_dir:
        reference = load_classical_reference(run_dir)
        if reference is None:
            continue
        name = f"classical_{run_dir.name}"
        matrices_for_alignment[name] = (reference["track_ids"], reference["matrix"])

    aligned, aligned_track_ids = align_matrices_by_track_id(matrices_for_alignment, reference_name="classical_full")
    classical_full = aligned.pop("classical_full")
    representation_matrices = {name: matrix for name, matrix in aligned.items() if name in rep_manifests}
    all_matrices = {"classical_full": classical_full, **aligned}

    for name, matrix in all_matrices.items():
        validate_matrix(name, matrix)

    inter_rows = pairwise_cka_table(representation_matrices, method="linear", seed=args.seed)
    inter_rbf_rows = pairwise_cka_table(representation_matrices, method="rbf", rbf_sample_size=args.rbf_sample_size, seed=args.seed)
    save_csv(output_dir / "inter_architecture_linear_cka.csv", inter_rows)
    save_csv(output_dir / "inter_architecture_rbf_cka.csv", inter_rbf_rows)
    labels, matrix = matrix_table(inter_rows)
    save_matrix_csv(output_dir / "inter_architecture_linear_cka_matrix.csv", labels, matrix)

    classical_rows = []
    for name, matrix_value in representation_matrices.items():
        classical_rows.append({"representation": name, "reference": "classical_full", "linear_cka": float(linear_cka(matrix_value, classical_full))})
        classical_rows.append(
            {
                "representation": name,
                "reference": "classical_full",
                "rbf_cka": float(rbf_cka(matrix_value, classical_full, sample_size=args.rbf_sample_size, seed=args.seed)),
                "rbf_sample_size": args.rbf_sample_size,
            }
        )
    save_csv(output_dir / "full_classical_feature_cka.csv", classical_rows)

    group_profile_rows, group_profile_rbf_rows = compute_group_profiles(
        representation_matrices=representation_matrices,
        feature_matrix=classical_full,
        group_indices=group_indices,
        rbf_sample_size=args.rbf_sample_size,
        seed=args.seed,
    )
    save_csv(output_dir / "feature_group_linear_cka_profiles.csv", group_profile_rows)
    save_csv(output_dir / "feature_group_rbf_cka_profiles.csv", group_profile_rbf_rows)

    group_group_rows = pairwise_cka_table({name: classical_full[:, indices] for name, indices in group_indices.items()}, method="linear", seed=args.seed)
    save_csv(output_dir / "feature_group_collinearity_linear_cka.csv", group_group_rows)
    group_labels, group_matrix = matrix_table(group_group_rows)
    save_matrix_csv(output_dir / "feature_group_collinearity_linear_cka_matrix.csv", group_labels, group_matrix)

    metrics = {name: metric_from_run({"path": rep_paths[name], "manifest": rep_manifests[name]}) for name in representation_matrices}
    novelty_rows = compute_novelty_gap(group_profile_rows, metrics=metrics)
    save_csv(output_dir / "novelty_gap.csv", novelty_rows)

    correlation_output = []
    for metric_name in ("accuracy", "macro_f1", "balanced_accuracy"):
        correlation_output.append(
            correlation_rows(
                np.asarray([row["novelty_gap_vs_best_group"] for row in novelty_rows], dtype=np.float64),
                np.asarray([np.nan if row[metric_name] is None else row[metric_name] for row in novelty_rows], dtype=np.float64),
                label=f"novelty_gap_vs_{metric_name}",
            )
        )
        correlation_output.append(
            correlation_rows(
                np.asarray([row["full_feature_linear_cka"] for row in novelty_rows], dtype=np.float64),
                np.asarray([np.nan if row[metric_name] is None else row[metric_name] for row in novelty_rows], dtype=np.float64),
                label=f"full_feature_cka_vs_{metric_name}",
            )
        )
    save_csv(output_dir / "cka_vs_performance_correlation.csv", correlation_output)

    ceiling_rows = within_architecture_rows(list(representation_matrices), rep_manifests, representation_matrices)
    save_csv(output_dir / "within_architecture_cross_seed_ceiling.csv", ceiling_rows)

    untrained_rows: list[dict[str, Any]] = []
    if args.compute_untrained_floor:
        for name, matrix_value in representation_matrices.items():
            model_name = rep_manifests.get(name, {}).get("model")
            if model_name is None:
                untrained_rows.append({"representation": name, "status": "skipped_missing_model_in_manifest", "linear_cka": None})
                continue
            untrained = extract_untrained_representations(
                model_name=str(model_name),
                split=args.split,
                track_ids=aligned_track_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device,
                seed=args.seed,
            )
            untrained_rows.append({"representation": name, "model": model_name, "status": "computed", "linear_cka": float(linear_cka(matrix_value, untrained))})
    else:
        untrained_rows.append({"status": "skipped_pass_--compute-untrained-floor_to_run"})
    save_csv(output_dir / "untrained_floor.csv", untrained_rows)

    manifest = {
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "split": args.split,
        "sample_count": int(aligned_track_ids.shape[0]),
        "representation_dirs": [str(path.resolve()) for path in args.representation_dir],
        "classical_run_dirs": [str(path.resolve()) for path in args.classical_run_dir],
        "outputs": sorted(path.name for path in output_dir.iterdir()),
        "notes": ["untrained floor is computed only when --compute-untrained-floor is passed"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
