from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


ARCHES = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")


@dataclass(frozen=True)
class Representation:
    arch: str
    path: Path
    matrix: np.ndarray
    labels: np.ndarray
    track_ids: np.ndarray
    manifest: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_hub_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "results" / "representations").exists() and (candidate / "music").exists():
            return candidate
    raise RuntimeError(f"could not locate hub root from {start}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_representation(path: Path) -> Representation:
    matrix = np.load(path / "penultimate.npy").astype(np.float64, copy=False)
    labels = np.load(path / "labels.npy").astype(np.int64, copy=False)
    track_ids = np.load(path / "track_ids.npy").astype(np.int64, copy=False)
    manifest = read_json(path / "manifest.json")
    arch = str(manifest.get("model") or path.name.split("_seed")[0])
    if arch == "lstm":
        arch = "bilstm"
    if matrix.ndim != 2:
        raise ValueError(f"{path}: penultimate.npy must be 2D, found {matrix.shape}")
    if matrix.shape[0] != labels.shape[0] or labels.shape[0] != track_ids.shape[0]:
        raise ValueError(f"{path}: matrix, labels, and track_ids are not aligned")
    if np.unique(track_ids).shape[0] != track_ids.shape[0]:
        raise ValueError(f"{path}: duplicate track_ids")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: non-finite representation values")
    return Representation(arch=arch, path=path, matrix=matrix, labels=labels, track_ids=track_ids, manifest=manifest)


def align_representations(reps: list[Representation]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    maps = {rep.arch: {int(track_id): i for i, track_id in enumerate(rep.track_ids.tolist())} for rep in reps}
    common_ids = set.intersection(*(set(track_map) for track_map in maps.values()))
    if not common_ids:
        raise ValueError("no common track_ids across representations")
    reference = reps[0]
    ordered_ids = np.asarray([int(track_id) for track_id in reference.track_ids.tolist() if int(track_id) in common_ids], dtype=np.int64)
    aligned: dict[str, np.ndarray] = {}
    aligned_labels: np.ndarray | None = None
    for rep in reps:
        indices = [maps[rep.arch][int(track_id)] for track_id in ordered_ids.tolist()]
        aligned[rep.arch] = rep.matrix[indices]
        labels = rep.labels[indices]
        if aligned_labels is None:
            aligned_labels = labels
        elif not np.array_equal(aligned_labels, labels):
            raise ValueError(f"{rep.arch}: labels disagree after track_id alignment")
    assert aligned_labels is not None
    return aligned, aligned_labels, ordered_ids


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_z = scaler.fit_transform(train)
    test_z = scaler.transform(test)
    return train_z, test_z, scaler


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, max_iter: int, c_value: float) -> LogisticRegression:
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        n_jobs=None,
        random_state=0,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    return clf


def ridge_bridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    """Map standardized target reps into standardized source-rep coordinates."""
    xtx = x_train.T @ x_train
    regularizer = alpha * np.eye(xtx.shape[0], dtype=np.float64)
    weights = np.linalg.solve(xtx + regularizer, x_train.T @ y_train)
    return x_test @ weights


def metric_row(source: str, target: str, mode: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "source_arch": source,
        "target_arch": target,
        "mode": mode,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    transfer_rows = [row for row in rows if row["mode"] == "source_probe_on_target_via_ridge_bridge" and row["source_arch"] != row["target_arch"]]
    self_rows = [row for row in rows if row["mode"] == "self_probe"]
    by_source: dict[str, list[float]] = {}
    by_target: dict[str, list[float]] = {}
    for row in transfer_rows:
        by_source.setdefault(str(row["source_arch"]), []).append(float(row["macro_f1"]))
        by_target.setdefault(str(row["target_arch"]), []).append(float(row["macro_f1"]))
    self_by_arch = {str(row["source_arch"]): float(row["macro_f1"]) for row in self_rows}
    mean_transfer = float(np.mean([row["macro_f1"] for row in transfer_rows]))
    mean_self = float(np.mean([row["macro_f1"] for row in self_rows]))
    return {
        "mean_offdiag_transfer_macro_f1": mean_transfer,
        "mean_self_probe_macro_f1": mean_self,
        "mean_transfer_retention_vs_self": float(mean_transfer / mean_self) if mean_self else None,
        "best_offdiag_transfer": max(transfer_rows, key=lambda row: float(row["macro_f1"])),
        "worst_offdiag_transfer": min(transfer_rows, key=lambda row: float(row["macro_f1"])),
        "mean_transfer_macro_f1_by_source": {arch: float(np.mean(values)) for arch, values in sorted(by_source.items())},
        "mean_transfer_macro_f1_by_target": {arch: float(np.mean(values)) for arch, values in sorted(by_target.items())},
        "self_probe_macro_f1_by_arch": self_by_arch,
    }


def write_report(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    best = summary["best_offdiag_transfer"]
    worst = summary["worst_offdiag_transfer"]
    lines = [
        "# Cross-Architecture Linear Probe Transfer",
        "",
        f"Created: {manifest['created_at_utc']}",
        "",
        "Protocol: seed-42 penultimate test representations were aligned by `track_ids`, split once with a stratified 70/30 internal train/test split, and evaluated with class-balanced multinomial logistic probes. For off-diagonal transfer, a ridge linear bridge is fit on the internal train fold from target representation coordinates into source representation coordinates; the source probe is then evaluated on bridged target test representations.",
        "",
        "This is a supervised representation-transfer check, not a replacement for the frozen headline metrics. It uses only local seed-42 representation arrays because seeds 43-46 are pointer-backed on WinPC in this hub.",
        "",
        "## Headline",
        "",
        f"- Mean self-probe macro F1: {summary['mean_self_probe_macro_f1']:.4f}",
        f"- Mean off-diagonal bridged-transfer macro F1: {summary['mean_offdiag_transfer_macro_f1']:.4f}",
        f"- Transfer retention vs self-probe mean: {summary['mean_transfer_retention_vs_self']:.3f}",
        f"- Best off-diagonal transfer: {best['source_arch']} -> {best['target_arch']} macro F1 {best['macro_f1']:.4f}",
        f"- Worst off-diagonal transfer: {worst['source_arch']} -> {worst['target_arch']} macro F1 {worst['macro_f1']:.4f}",
        "",
        "## Output Files",
        "",
        "- `pairwise_transfer.csv`: self and off-diagonal probe-transfer metrics.",
        "- `summary.json`: machine-readable headline summary and provenance.",
        "- `manifest.json`: command, parameters, input paths, and output inventory.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    default_root = find_hub_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(description="Run cross-architecture linear probe transfer on local representation arrays.")
    parser.add_argument("--hub-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    hub_root = args.hub_root.resolve()
    output_dir = (args.output_dir or (hub_root / "results" / "linear_probe_transfer")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rep_dirs = [hub_root / "results" / "representations" / f"{arch}_seed42" for arch in ARCHES]
    reps = [load_representation(path) for path in rep_dirs]
    matrices, labels, track_ids = align_representations(reps)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(np.zeros(labels.shape[0]), labels))
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    standardized: dict[str, dict[str, np.ndarray]] = {}
    probes: dict[str, LogisticRegression] = {}
    rows: list[dict[str, Any]] = []

    for arch, matrix in matrices.items():
        x_train_z, x_test_z, _scaler = standardize(matrix[train_idx], matrix[test_idx])
        standardized[arch] = {"train": x_train_z, "test": x_test_z}
        probes[arch] = fit_probe(x_train_z, y_train, max_iter=args.max_iter, c_value=args.c)
        rows.append(metric_row(arch, arch, "self_probe", y_test, probes[arch].predict(x_test_z)))

    for source in ARCHES:
        for target in ARCHES:
            if source == target:
                continue
            bridged_target_test = ridge_bridge(
                standardized[target]["train"],
                standardized[source]["train"],
                standardized[target]["test"],
                alpha=args.ridge_alpha,
            )
            rows.append(
                metric_row(
                    source,
                    target,
                    "source_probe_on_target_via_ridge_bridge",
                    y_test,
                    probes[source].predict(bridged_target_test),
                )
            )

    summary = summarize(rows)
    manifest = {
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "protocol": "seed42_test_representations_internal_stratified_split_ridge_bridged_probe_transfer",
        "sample_count_aligned": int(labels.shape[0]),
        "train_count": int(train_idx.shape[0]),
        "test_count": int(test_idx.shape[0]),
        "n_classes": int(np.unique(labels).shape[0]),
        "split_seed": args.seed,
        "test_size": args.test_size,
        "ridge_alpha": args.ridge_alpha,
        "logistic_regression": {"C": args.c, "class_weight": "balanced", "max_iter": args.max_iter, "solver": "lbfgs"},
        "input_representation_dirs": [str(path.resolve()) for path in rep_dirs],
        "track_id_sha256_note": "track IDs are stored in aligned_track_ids.npy",
        "outputs": ["aligned_track_ids.npy", "manifest.json", "pairwise_transfer.csv", "report.md", "summary.json"],
        "limitations": [
            "Uses local seed-42 test representations with an internal split; training-split representations are not local in this hub.",
            "Off-diagonal transfer uses a ridge linear bridge because raw probe coefficients are not coordinate-aligned across independently trained architectures and representation dimensions differ.",
        ],
    }

    np.save(output_dir / "aligned_track_ids.npy", track_ids, allow_pickle=False)
    write_csv(output_dir / "pairwise_transfer.csv", rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", summary, manifest)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
