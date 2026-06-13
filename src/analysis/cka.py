from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.data.features import FEATURE_OUTPUT_DIR, load_joined_feature_table


EXPECTED_GROUP_WIDTHS = {
    "timbre": 140,
    "pitch_class": 84,
    "tonal_geometry": 42,
    "spectral_contrast": 49,
    "spectral_shape": 21,
    "noisiness": 7,
    "energy": 7,
    "rhythm": 22,
}


def validate_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, found shape {array.shape}")
    if array.shape[0] < 2:
        raise ValueError(f"{name} needs at least two samples for CKA")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array.astype(np.float64, copy=False)


# double-center the gram matrix so cka measures similarity of the centered
# kernels. without this step a shared additive offset would inflate similarity.
def center_gram(gram: np.ndarray) -> np.ndarray:
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"gram matrix must be square, found {gram.shape}")
    return gram - gram.mean(axis=0, keepdims=True) - gram.mean(axis=1, keepdims=True) + gram.mean()


def _safe_cka_from_grams(k: np.ndarray, l: np.ndarray) -> float:
    kc = center_gram(k)
    lc = center_gram(l)
    numerator = float(np.sum(kc * lc))
    denominator = float(np.sqrt(np.sum(kc * kc) * np.sum(lc * lc)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    value = numerator / denominator
    if not np.isfinite(value):
        raise ValueError("CKA produced a non-finite value")
    return float(np.clip(value, -1.0, 1.0))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = validate_matrix("x", x)
    y = validate_matrix("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}")
    k = x @ x.T
    l = y @ y.T
    return _safe_cka_from_grams(k, l)


def pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    x = validate_matrix("x", x)
    squared_norms = np.sum(x * x, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (x @ x.T)
    return np.maximum(distances, 0.0)


# we set the rbf bandwidth from the median pairwise distance so the kernel
# adapts to each representation's scale instead of a hand-tuned gamma.
def median_heuristic_gamma(x: np.ndarray) -> float:
    distances = pairwise_squared_distances(x)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        raise ValueError("cannot compute RBF median heuristic for identical samples")
    median = float(np.median(positive))
    if median <= 0.0 or not np.isfinite(median):
        raise ValueError("invalid RBF median distance")
    return 1.0 / (2.0 * median)


def rbf_kernel(x: np.ndarray, gamma: float | None = None) -> np.ndarray:
    x = validate_matrix("x", x)
    if gamma is None:
        gamma = median_heuristic_gamma(x)
    if gamma <= 0.0 or not np.isfinite(gamma):
        raise ValueError(f"invalid RBF gamma: {gamma}")
    return np.exp(-gamma * pairwise_squared_distances(x))


def sampled_indices(n_samples: int, sample_size: int | None, seed: int) -> np.ndarray:
    if sample_size is None or sample_size >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=sample_size, replace=False).astype(np.int64))


# the rbf path builds n-by-n grams, so we subsample to a fixed size with a
# seeded rng to keep memory bounded and the estimate reproducible across runs.
def rbf_cka(x: np.ndarray, y: np.ndarray, sample_size: int | None = 1024, seed: int = 0) -> float:
    x = validate_matrix("x", x)
    y = validate_matrix("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}")
    indices = sampled_indices(x.shape[0], sample_size, seed)
    xs = x[indices]
    ys = y[indices]
    return _safe_cka_from_grams(rbf_kernel(xs), rbf_kernel(ys))


def assert_cka_sanity(matrix: np.ndarray, atol: float = 1e-8) -> dict[str, float]:
    matrix = validate_matrix("matrix", matrix)
    identity = linear_cka(matrix, matrix)
    if not np.isclose(identity, 1.0, atol=atol):
        raise AssertionError(f"linear CKA identity check failed: {identity}")
    split_a = matrix[:, ::2] if matrix.shape[1] > 1 else matrix
    split_b = matrix[:, 1::2] if matrix.shape[1] > 1 else matrix
    symmetry_ab = linear_cka(split_a, split_b)
    symmetry_ba = linear_cka(split_b, split_a)
    if not np.isclose(symmetry_ab, symmetry_ba, atol=atol):
        raise AssertionError(f"linear CKA symmetry check failed: {symmetry_ab} != {symmetry_ba}")
    return {"identity": float(identity), "symmetry_ab": float(symmetry_ab), "symmetry_ba": float(symmetry_ba)}


def _index_map(track_ids: np.ndarray, name: str) -> dict[int, int]:
    ids = np.asarray(track_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError(f"{name} track_ids must be 1D")
    if np.unique(ids).shape[0] != ids.shape[0]:
        raise ValueError(f"{name} track_ids contain duplicates")
    return {int(track_id): int(index) for index, track_id in enumerate(ids.tolist())}


def align_matrices_by_track_id(
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_name: str | None = None,
    require_same_order: bool = False,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not matrices:
        raise ValueError("no matrices were provided for alignment")
    maps = {name: _index_map(track_ids, name) for name, (track_ids, _matrix) in matrices.items()}
    common_ids = set.intersection(*(set(item.keys()) for item in maps.values()))
    if not common_ids:
        raise ValueError("matrices have no overlapping track_ids")

    if reference_name is None:
        reference_name = next(iter(matrices))
    reference_ids = np.asarray(matrices[reference_name][0], dtype=np.int64)
    ordered_ids = np.asarray([int(track_id) for track_id in reference_ids.tolist() if int(track_id) in common_ids], dtype=np.int64)
    if ordered_ids.shape[0] != len(common_ids):
        raise ValueError("reference track_ids did not cover the full common set")

    aligned: dict[str, np.ndarray] = {}
    for name, (_track_ids, matrix) in matrices.items():
        if require_same_order and not np.array_equal(np.asarray(_track_ids, dtype=np.int64), reference_ids):
            raise ValueError(f"{name} track_ids are not in the same order as {reference_name}")
        indices = [maps[name][int(track_id)] for track_id in ordered_ids.tolist()]
        matrix_array = validate_matrix(name, np.asarray(matrix))
        aligned[name] = matrix_array[indices]
    return aligned, ordered_ids


def require_identical_track_ids(reference: np.ndarray, candidate: np.ndarray, name: str = "candidate") -> None:
    if not np.array_equal(np.asarray(reference, dtype=np.int64), np.asarray(candidate, dtype=np.int64)):
        raise ValueError(f"{name} track_ids are not exactly aligned")


def load_feature_group_payload(path: Path | None = None) -> dict[str, list[str]]:
    payload_path = path or (FEATURE_OUTPUT_DIR / "feature_groups.json")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    groups = payload.get("feature_groups", payload)
    if not isinstance(groups, dict):
        raise ValueError(f"invalid feature group payload: {payload_path}")
    return {str(name): [str(column) for column in columns] for name, columns in groups.items()}


def resolve_feature_group_indices(path: Path | None = None) -> dict[str, list[int]]:
    full_matrix, _derived_groups = load_joined_feature_table()
    feature_groups = load_feature_group_payload(path)
    column_to_index = {str(column): int(index) for index, column in enumerate(full_matrix.columns.tolist())}
    indices: dict[str, list[int]] = {}
    seen: set[int] = set()
    for name, expected_width in EXPECTED_GROUP_WIDTHS.items():
        if name not in feature_groups:
            raise ValueError(f"feature_groups.json is missing group {name}")
        missing_columns = [column for column in feature_groups[name] if column not in column_to_index]
        if missing_columns:
            raise ValueError(f"group {name} has columns absent from joined feature table: {missing_columns[:5]}")
        group_indices = [column_to_index[column] for column in feature_groups[name]]
        if len(group_indices) != expected_width:
            raise ValueError(f"group {name} expected width {expected_width}, found {len(group_indices)}")
        duplicates = seen.intersection(group_indices)
        if duplicates:
            raise ValueError(f"duplicate feature indices across groups: {sorted(duplicates)[:10]}")
        seen.update(group_indices)
        indices[name] = group_indices
    total = sum(len(value) for value in indices.values())
    if total != 372:
        raise ValueError(f"expected 372 grouped CKA dimensions, found {total}")
    return indices


def write_feature_index_manifest(output_path: Path) -> dict[str, Any]:
    full_matrix, _derived_groups = load_joined_feature_table()
    indices = resolve_feature_group_indices()
    payload = {
        "full_feature_width": int(full_matrix.shape[1]),
        "cka_group_width": int(sum(len(value) for value in indices.values())),
        "groups": {
            name: {
                "width": len(group_indices),
                "indices": group_indices,
                "columns": [str(full_matrix.columns[index]) for index in group_indices],
            }
            for name, group_indices in indices.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_split_features(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features_path = FEATURE_OUTPUT_DIR / f"{split}_features.npy"
    labels_path = FEATURE_OUTPUT_DIR / f"{split}_labels.npy"
    track_ids_path = FEATURE_OUTPUT_DIR / f"{split}_track_ids.npy"
    if not features_path.exists() and split == "train":
        features_path = FEATURE_OUTPUT_DIR / "training_features.npy"
        labels_path = FEATURE_OUTPUT_DIR / "training_labels.npy"
        track_ids_path = FEATURE_OUTPUT_DIR / "training_track_ids.npy"
    features = np.load(features_path)
    labels = np.load(labels_path)
    track_ids = np.load(track_ids_path).astype(np.int64)
    validate_matrix(f"{split}_features", features)
    if labels.shape[0] != track_ids.shape[0] or features.shape[0] != track_ids.shape[0]:
        raise ValueError(f"{split} feature artifacts are not aligned")
    return features.astype(np.float64, copy=False), labels.astype(np.int64), track_ids


def pairwise_cka_table(named_matrices: dict[str, np.ndarray], method: str = "linear", rbf_sample_size: int | None = 1024, seed: int = 0) -> list[dict[str, Any]]:
    names = list(named_matrices)
    rows: list[dict[str, Any]] = []
    for i, left_name in enumerate(names):
        for right_name in names[i:]:
            if method == "linear":
                value = linear_cka(named_matrices[left_name], named_matrices[right_name])
            elif method == "rbf":
                value = rbf_cka(named_matrices[left_name], named_matrices[right_name], sample_size=rbf_sample_size, seed=seed)
            else:
                raise ValueError(f"unsupported CKA method: {method}")
            rows.append({"left": left_name, "right": right_name, "method": method, "cka": float(value)})
    return rows


def matrix_table(rows: Iterable[dict[str, Any]], value_key: str = "cka") -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    for row in rows:
        for key in ("left", "right"):
            name = str(row[key])
            if name not in labels:
                labels.append(name)
    index = {name: i for i, name in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    for row in rows:
        i = index[str(row["left"])]
        j = index[str(row["right"])]
        matrix[i, j] = float(row[value_key])
        matrix[j, i] = float(row[value_key])
    return labels, matrix


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

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


def save_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[float(value) if np.isfinite(value) else "" for value in row]])


def correlation_rows(x: np.ndarray, y: np.ndarray, label: str) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"label": label, "n": int(mask.sum()), "pearson_r": None, "spearman_rho": None, "spearman_p": None}
    xm = x[mask]
    ym = y[mask]
    pearson = float(np.corrcoef(xm, ym)[0, 1])
    try:
        from scipy.stats import spearmanr

        result = spearmanr(xm, ym)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    except Exception:
        xr = np.argsort(np.argsort(xm)).astype(np.float64)
        yr = np.argsort(np.argsort(ym)).astype(np.float64)
        rho = float(np.corrcoef(xr, yr)[0, 1])
        pvalue = None
    return {"label": label, "n": int(mask.sum()), "pearson_r": pearson, "spearman_rho": rho, "spearman_p": pvalue}


def run_smoke_checks() -> dict[str, Any]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(32, 12))
    y = x @ rng.normal(size=(12, 8)) + 0.01 * rng.normal(size=(32, 8))
    identity = linear_cka(x, x)
    symmetry_xy = linear_cka(x, y)
    symmetry_yx = linear_cka(y, x)
    rbf_identity = rbf_cka(x, x, sample_size=16, seed=1)
    if not np.isclose(identity, 1.0, atol=1e-10):
        raise AssertionError(f"linear identity failed: {identity}")
    if not np.isclose(rbf_identity, 1.0, atol=1e-10):
        raise AssertionError(f"rbf identity failed: {rbf_identity}")
    if not np.isclose(symmetry_xy, symmetry_yx, atol=1e-10):
        raise AssertionError("linear symmetry failed")
    require_identical_track_ids(np.arange(5), np.arange(5), name="same")
    try:
        require_identical_track_ids(np.arange(5), np.array([0, 2, 1, 3, 4]), name="shuffled")
    except ValueError:
        shuffled_failed = True
    else:
        shuffled_failed = False
    if not shuffled_failed:
        raise AssertionError("shuffled track ID check did not fail")
    group_indices = resolve_feature_group_indices()
    group_width_total = sum(len(indices) for indices in group_indices.values())
    if group_width_total != 372:
        raise AssertionError(f"group widths total {group_width_total}, expected 372")
    return {
        "linear_identity": float(identity),
        "rbf_identity": float(rbf_identity),
        "symmetry_xy": float(symmetry_xy),
        "symmetry_yx": float(symmetry_yx),
        "shuffled_track_ids_failed": shuffled_failed,
        "group_width_total": group_width_total,
        "finite": True,
    }
