from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_READY_DIR = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready"
SPLITS_DIR = REPO_ROOT / "data" / "splits" / "fma_medium"
RAW_METADATA_DIR = REPO_ROOT / "data" / "raw" / "fma_metadata"
FEATURE_OUTPUT_DIR = REPO_ROOT / "data" / "features" / "fma_medium"

MANIFEST_PATH = BENCHMARK_READY_DIR / "manifest.csv"
LABEL_MAP_PATH = SPLITS_DIR / "label_map.json"
CLASS_WEIGHTS_PATH = SPLITS_DIR / "class_weights.json"
FMA_FEATURES_PATH = RAW_METADATA_DIR / "features.csv"
RHYTHM_FEATURES_PATH = RAW_METADATA_DIR / "rhythm_features.csv"

RHYTHM_FEATURE_NAMES = [
    "tempo_bpm",
    "beat_count",
    "beat_density",
    "beat_strength_mean",
    "beat_strength_std",
    "beat_strength_max",
    "onset_strength_mean",
    "onset_strength_std",
    "onset_strength_max",
    "onset_strength_skew",
    "onset_strength_kurtosis",
    "ibi_mean_seconds",
    "ibi_std_seconds",
    "ibi_min_seconds",
    "ibi_max_seconds",
    "ibi_median_seconds",
    "tempogram_energy_mean",
    "tempogram_energy_std",
    "tempogram_energy_max",
    "tempogram_peak_1",
    "tempogram_peak_2",
    "tempogram_peak_3",
]


# manifest and split helpers


def load_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest["track_id"] = manifest["track_id"].astype(int)
    return manifest


def read_split_ids(split: str) -> np.ndarray:
    split_path = BENCHMARK_READY_DIR / f"{split}_ids.txt"
    split_ids = np.loadtxt(split_path, dtype=np.int64)
    return np.atleast_1d(split_ids)


# fma's features.csv uses a three-level multiindex header (family, stat, coeff).
# flatten it into a single pipe-delimited string so pandas gives us a flat
# column index that is easy to filter with string matching.
def flatten_feature_column(column: tuple[str, str, str]) -> str:
    family, statistic, coefficient = column
    return f"{family}|{statistic}|{coefficient}"


def load_fma_feature_table() -> pd.DataFrame:
    features = pd.read_csv(FMA_FEATURES_PATH, header=[0, 1, 2], index_col=0)
    features.index = features.index.astype(int)
    features.columns = [flatten_feature_column(tuple(column)) for column in features.columns]
    return features


# map each of the 8 canonical cka groups to the column names that belong to it.
# the 8 groups cover 372 of the 540 total features; the remaining 168 columns
# (chroma_stft and spectral_flatness variants) are in the classical matrix but
# excluded from cka because they overlap with existing groups.
def build_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "timbre": [name for name in feature_columns if name.startswith("mfcc|")],
        "pitch_class": [name for name in feature_columns if name.startswith("chroma_cens|")],
        "tonal_geometry": [name for name in feature_columns if name.startswith("tonnetz|")],
        "spectral_contrast": [name for name in feature_columns if name.startswith("spectral_contrast|")],
        "spectral_shape": [
            name
            for name in feature_columns
            if name.startswith("spectral_centroid|")
            or name.startswith("spectral_bandwidth|")
            or name.startswith("spectral_rolloff|")
        ],
        "noisiness": [name for name in feature_columns if name.startswith("zcr|")],
        "energy": [name for name in feature_columns if name.startswith("rmse|")],
        "rhythm": [name for name in RHYTHM_FEATURE_NAMES if name in feature_columns],
    }
    return groups


# hard check: the joined matrix must be exactly 540 columns (518 fma + 22 rhythm)
# and the 8 cka groups must sum to exactly 372 dimensions. if either is wrong,
# downstream models and the cka analysis would silently use the wrong features.
def assert_feature_layout(full_matrix: pd.DataFrame, feature_groups: dict[str, list[str]]) -> None:
    full_width = full_matrix.shape[1]
    cka_width = sum(len(columns) for columns in feature_groups.values())
    assert full_width == 540, f"expected 540 feature columns, found {full_width}"
    assert cka_width == 372, f"expected 372 cka columns, found {cka_width}"


# create the canonical label map (alphabetical, 0-indexed) and per-class
# inverse-frequency weights from the training split only.
def write_label_artifacts() -> tuple[dict[str, int], dict[int, str], dict[int, float]]:
    manifest = load_manifest()
    training_manifest = manifest.loc[manifest["split"] == "training"].copy()

    # alphabetical sort gives a deterministic, reproducible encoding
    classes = sorted(training_manifest["genre_top"].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(classes)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    # raw inverse-frequency: rare classes get higher weight. consumers that need
    # normalized weights (e.g. pytorch cross-entropy) should divide by the mean.
    counts = training_manifest["genre_top"].value_counts().sort_index()
    inverse_frequency = {label_to_id[label]: float(1.0 / count) for label, count in counts.items()}

    label_payload = {
        "classes": classes,
        "label_to_id": label_to_id,
        "id_to_label": {str(index): label for index, label in id_to_label.items()},
        "source_manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
    }
    LABEL_MAP_PATH.write_text(json.dumps(label_payload, indent=2), encoding="utf-8")

    weights_payload = {
        "weights_by_id": {str(index): weight for index, weight in inverse_frequency.items()},
        "weights_by_label": {label: inverse_frequency[label_to_id[label]] for label in classes},
        "counts_by_label": {label: int(counts[label]) for label in classes},
        "source_split": "training",
    }
    CLASS_WEIGHTS_PATH.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")
    return label_to_id, id_to_label, inverse_frequency


def load_label_map() -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    label_to_id = {str(label): int(index) for label, index in payload["label_to_id"].items()}
    id_to_label = {int(index): str(label) for index, label in payload["id_to_label"].items()}
    return label_to_id, id_to_label


# join the 518-column fma feature table with the 22 custom rhythm features
# to produce the full 540-d classical matrix. also builds the 8-group cka
# column mapping and validates both dimension targets.
def load_joined_feature_table() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    manifest = load_manifest()
    benchmark_ids = manifest["track_id"].tolist()

    fma_features = load_fma_feature_table()
    missing_fma_ids = sorted(set(benchmark_ids) - set(fma_features.index.tolist()))
    if missing_fma_ids:
        raise ValueError(f"missing {len(missing_fma_ids)} benchmark ids in features.csv")

    rhythm_features = pd.read_csv(RHYTHM_FEATURES_PATH)
    rhythm_features["track_id"] = rhythm_features["track_id"].astype(int)
    rhythm_features = rhythm_features.set_index("track_id")

    missing_rhythm_ids = sorted(set(benchmark_ids) - set(rhythm_features.index.tolist()))
    if missing_rhythm_ids:
        raise ValueError(f"missing {len(missing_rhythm_ids)} benchmark ids in rhythm_features.csv")

    filtered_fma = fma_features.loc[benchmark_ids].copy()
    filtered_rhythm = rhythm_features.loc[benchmark_ids, RHYTHM_FEATURE_NAMES].copy()
    full_matrix = filtered_fma.join(filtered_rhythm, how="inner")

    feature_groups = build_feature_groups(full_matrix.columns.tolist())
    assert_feature_layout(full_matrix, feature_groups)
    return full_matrix, feature_groups


# build per-split (N, 540) numpy arrays, fit a standard scaler on training
# data only, and persist everything to data/features/fma_medium/.
def build_feature_artifacts() -> dict[str, object]:
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    label_to_id, id_to_label = load_label_map()
    full_matrix, feature_groups = load_joined_feature_table()

    split_outputs: dict[str, dict[str, object]] = {}
    train_ids = read_split_ids("training")
    train_matrix = full_matrix.loc[train_ids].to_numpy(dtype=np.float32, copy=True)

    # fit scaler on training only to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_matrix)
    joblib.dump(scaler, FEATURE_OUTPUT_DIR / "scaler.joblib")

    for split in ["training", "validation", "test"]:
        split_ids = read_split_ids(split)
        split_manifest = manifest.loc[manifest["track_id"].isin(split_ids)].copy()
        split_manifest = split_manifest.set_index("track_id").loc[split_ids].reset_index()

        raw_matrix = full_matrix.loc[split_ids].to_numpy(dtype=np.float32, copy=True)
        scaled_matrix = scaler.transform(raw_matrix).astype(np.float32, copy=False)
        labels = split_manifest["genre_top"].map(label_to_id).to_numpy(dtype=np.int64)
        track_ids = split_manifest["track_id"].to_numpy(dtype=np.int64)

        np.save(FEATURE_OUTPUT_DIR / f"{split}_features.npy", scaled_matrix, allow_pickle=False)
        np.save(FEATURE_OUTPUT_DIR / f"{split}_track_ids.npy", track_ids, allow_pickle=False)
        np.save(FEATURE_OUTPUT_DIR / f"{split}_labels.npy", labels, allow_pickle=False)

        split_outputs[split] = {
            "shape": list(scaled_matrix.shape),
            "class_counts": {label: int(count) for label, count in split_manifest["genre_top"].value_counts().sort_index().items()},
            "nan_count": int(np.isnan(scaled_matrix).sum()),
            "inf_count": int(np.isinf(scaled_matrix).sum()),
        }

    feature_groups_payload = {
        "feature_groups": feature_groups,
        "cka_group_width": sum(len(columns) for columns in feature_groups.values()),
        "full_feature_width": int(full_matrix.shape[1]),
    }
    (FEATURE_OUTPUT_DIR / "feature_groups.json").write_text(json.dumps(feature_groups_payload, indent=2), encoding="utf-8")

    summary = {
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "feature_source": str(FMA_FEATURES_PATH.relative_to(REPO_ROOT)),
        "rhythm_source": str(RHYTHM_FEATURES_PATH.relative_to(REPO_ROOT)),
        "label_map_path": str(LABEL_MAP_PATH.relative_to(REPO_ROOT)),
        "full_feature_width": int(full_matrix.shape[1]),
        "cka_group_width": sum(len(columns) for columns in feature_groups.values()),
        "splits": split_outputs,
        "label_round_trip_ok": all(id_to_label[index] in label_to_id for index in id_to_label),
    }
    (FEATURE_OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
