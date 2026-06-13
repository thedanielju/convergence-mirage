from __future__ import annotations

import argparse
import csv
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
    EXPECTED_GROUP_WIDTHS,
    correlation_rows,
    linear_cka,
    load_split_features,
    resolve_feature_group_indices,
    save_csv,
    save_matrix_csv,
)


LABEL_MAP_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def label_names() -> dict[int, str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {int(index): str(label) for index, label in payload["id_to_label"].items()}


def euclidean_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    squared = np.sum(centroids * centroids, axis=1, keepdims=True)
    distances = squared + squared.T - 2.0 * centroids @ centroids.T
    return np.sqrt(np.maximum(distances, 0.0))


def compute_group_collinearity(features: np.ndarray, group_indices: dict[str, list[int]]) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    names = list(group_indices)
    matrix = np.eye(len(names), dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for j, right in enumerate(names[i:], start=i):
            value = linear_cka(features[:, group_indices[left]], features[:, group_indices[right]])
            matrix[i, j] = value
            matrix[j, i] = value
            rows.append({"left_group": left, "right_group": right, "linear_cka": float(value)})
    return names, matrix, rows


def compute_centroids(features: np.ndarray, labels: np.ndarray, group_indices: dict[str, list[int]]) -> dict[str, dict[int, np.ndarray]]:
    centroids: dict[str, dict[int, np.ndarray]] = {}
    label_values = sorted(int(label) for label in np.unique(labels).tolist())
    for group_name, indices in group_indices.items():
        group_matrix = features[:, indices]
        centroids[group_name] = {}
        for label in label_values:
            mask = labels == label
            if not np.any(mask):
                continue
            centroids[group_name][label] = group_matrix[mask].mean(axis=0)
    return centroids


def centroid_rows(centroids: dict[str, dict[int, np.ndarray]], names: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, by_label in centroids.items():
        for label, vector in by_label.items():
            row = {
                "feature_group": group_name,
                "label_id": int(label),
                "label_name": names.get(int(label), str(label)),
                "width": int(vector.shape[0]),
                "l2_norm": float(np.linalg.norm(vector)),
            }
            for index, value in enumerate(vector.tolist()):
                row[f"dim_{index:03d}"] = float(value)
            rows.append(row)
    return rows


def pairwise_distance_rows(centroids: dict[str, dict[int, np.ndarray]], names: dict[int, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = sorted(names)
    for group_name, by_label in centroids.items():
        available = [label for label in labels if label in by_label]
        matrix = np.stack([by_label[label] for label in available], axis=0)
        distances = euclidean_distance_matrix(matrix)
        for i, left in enumerate(available):
            for j, right in enumerate(available):
                if i == j:
                    continue
                rows.append(
                    {
                        "feature_group": group_name,
                        "true_label": int(left),
                        "predicted_label": int(right),
                        "true_name": names.get(int(left), str(left)),
                        "predicted_name": names.get(int(right), str(right)),
                        "centroid_distance": float(distances[i, j]),
                    }
                )
    return rows


def load_run_confusions(run_dir: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    top_path = run_dir / "top_confusions.json"
    payload = read_json(top_path)
    if payload:
        rows = payload.get("top_confusions", payload if isinstance(payload, list) else [])
        if isinstance(rows, list):
            normalized = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                normalized.append(
                    {
                        "run_dir": str(run_dir),
                        "true_label": int(row["true_label"]),
                        "predicted_label": int(row["predicted_label"]),
                        "true_name": row.get("true_name", names.get(int(row["true_label"]), str(row["true_label"]))),
                        "predicted_name": row.get("predicted_name", names.get(int(row["predicted_label"]), str(row["predicted_label"]))),
                        "count": int(row.get("count", 0)),
                        "true_class_rate": float(row.get("true_class_rate", np.nan)),
                    }
                )
            return normalized

    confusion_path = run_dir / "confusion_matrix.npy"
    if not confusion_path.exists():
        return []
    confusion = np.load(confusion_path)
    row_totals = confusion.sum(axis=1, keepdims=True)
    rates = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=np.float64), where=row_totals != 0)
    rows = []
    for true_label in range(confusion.shape[0]):
        for predicted_label in range(confusion.shape[1]):
            if true_label == predicted_label or confusion[true_label, predicted_label] == 0:
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "true_name": names.get(true_label, str(true_label)),
                    "predicted_name": names.get(predicted_label, str(predicted_label)),
                    "count": int(confusion[true_label, predicted_label]),
                    "true_class_rate": float(rates[true_label, predicted_label]),
                }
            )
    return sorted(rows, key=lambda row: (row["true_class_rate"], row["count"]), reverse=True)[:20]


def join_confusions_with_distances(confusion_rows: list[dict[str, Any]], distance_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["feature_group"], int(row["true_label"]), int(row["predicted_label"])): float(row["centroid_distance"])
        for row in distance_rows
    }
    joined: list[dict[str, Any]] = []
    for row in confusion_rows:
        for group_name in EXPECTED_GROUP_WIDTHS:
            key = (group_name, int(row["true_label"]), int(row["predicted_label"]))
            joined.append({**row, "feature_group": group_name, "centroid_distance": lookup.get(key)})
    return joined


def all_pair_confusion_rows(run_dir: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    confusion_path = run_dir / "confusion_matrix.npy"
    if not confusion_path.exists():
        return []
    confusion = np.load(confusion_path)
    row_totals = confusion.sum(axis=1, keepdims=True)
    rates = np.divide(confusion, row_totals, out=np.zeros_like(confusion, dtype=np.float64), where=row_totals != 0)
    rows = []
    for true_label in range(confusion.shape[0]):
        for predicted_label in range(confusion.shape[1]):
            if true_label == predicted_label:
                continue
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "true_name": names.get(true_label, str(true_label)),
                    "predicted_name": names.get(predicted_label, str(predicted_label)),
                    "count": int(confusion[true_label, predicted_label]),
                    "true_class_rate": float(rates[true_label, predicted_label]),
                }
            )
    return rows


def h3_spearman_rows(run_dirs: list[Path], distance_rows: list[dict[str, Any]], names: dict[int, str]) -> list[dict[str, Any]]:
    distance_lookup = {
        (row["feature_group"], int(row["true_label"]), int(row["predicted_label"])): float(row["centroid_distance"])
        for row in distance_rows
    }
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        pair_rows = all_pair_confusion_rows(run_dir, names)
        if not pair_rows:
            rows.append({"run_dir": str(run_dir), "status": "skipped_no_confusion_matrix"})
            continue
        for group_name in EXPECTED_GROUP_WIDTHS:
            rates = []
            negative_distances = []
            counts = []
            for row in pair_rows:
                key = (group_name, int(row["true_label"]), int(row["predicted_label"]))
                distance = distance_lookup.get(key)
                if distance is None:
                    continue
                rates.append(float(row["true_class_rate"]))
                counts.append(float(row["count"]))
                # we negate distance so a positive correlation means closer genres get
                # confused more often, which is the direction h3 predicts.
                negative_distances.append(-float(distance))
            result = correlation_rows(np.asarray(negative_distances), np.asarray(rates), label=f"{run_dir.name}:{group_name}:negative_distance_vs_confusion_rate")
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "feature_group": group_name,
                    "hypothesis": "H3_confusions_increase_as_genre_centroid_distance_decreases",
                    "n_pairs": result["n"],
                    "spearman_rho": result["spearman_rho"],
                    "spearman_p": result["spearman_p"],
                    "pearson_r": result["pearson_r"],
                    "status": "computed" if result["n"] >= 3 else "skipped_insufficient_pairs",
                }
            )
    return rows


def write_large_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical feature error analysis for CKA hypotheses")
    parser.add_argument("--prediction-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (args.output_dir or (REPO_ROOT / "results" / "analysis" / f"error_analysis_{timestamp}")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    features, labels, track_ids = load_split_features(args.split)
    group_indices = resolve_feature_group_indices()
    names = label_names()

    group_names, collinearity_matrix, collinearity_rows = compute_group_collinearity(features, group_indices)
    save_csv(output_dir / "feature_group_collinearity_linear_cka.csv", collinearity_rows)
    save_matrix_csv(output_dir / "feature_group_collinearity_linear_cka_matrix.csv", group_names, collinearity_matrix)

    centroids = compute_centroids(features, labels, group_indices)
    write_large_csv(output_dir / "genre_centroids_by_group.csv", centroid_rows(centroids, names))
    distance_rows = pairwise_distance_rows(centroids, names)
    save_csv(output_dir / "pairwise_genre_distances_by_group.csv", distance_rows)

    top_confusions = []
    for run_dir in args.prediction_run_dir:
        top_confusions.extend(load_run_confusions(run_dir, names))
    joined = join_confusions_with_distances(top_confusions, distance_rows)
    save_csv(output_dir / "top_confusions_with_feature_distances.csv", joined)
    h3_rows = h3_spearman_rows(args.prediction_run_dir, distance_rows, names)
    save_csv(output_dir / "h3_spearman_tests.csv", h3_rows)

    manifest = {
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "split": args.split,
        "sample_count": int(track_ids.shape[0]),
        "feature_group_widths": {name: len(indices) for name, indices in group_indices.items()},
        "prediction_run_dirs": [str(path.resolve()) for path in args.prediction_run_dir],
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
