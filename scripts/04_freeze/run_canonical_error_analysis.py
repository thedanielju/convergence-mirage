from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.cka import resolve_feature_group_indices


MANIFEST_PATH = REPO_ROOT / "results" / "frozen_manifest.json"
AGG_CM_PATH = REPO_ROOT / "results" / "aggregated" / "per_model_aggregate_confusion.npz"
GROUPS_CKA_PATH = REPO_ROOT / "results" / "cka" / "arch_to_classical_groups.json"
LABEL_MAP_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json"
TRAIN_FEATURES_PATH = REPO_ROOT / "data" / "features" / "fma_medium" / "training_features.npy"
TRAIN_LABELS_PATH = REPO_ROOT / "data" / "features" / "fma_medium" / "training_labels.npy"
OUT_DIR = REPO_ROOT / "results" / "error_analysis"
DEEP_KEYS = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_label_names() -> dict[int, str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in payload["id_to_label"].items()}


def load_aggregate_confusions() -> dict[str, np.ndarray]:
    arch = np.load(AGG_CM_PATH, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    for key in arch.files:
        out[key] = arch[key]
    return out


def top_confused_pairs(cm: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
    n = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.where(row_sum > 0, cm / row_sum, 0.0)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # we score a genre pair by summing both off-diagonal directions so a confusion
            # that is asymmetric (a mistaken for b but not the reverse) still surfaces.
            sym = float(norm[i, j] + norm[j, i])
            if i < j:
                pairs.append((i, j, sym))
    pairs.sort(key=lambda t: t[2], reverse=True)
    seen: set[tuple[int, int]] = set()
    out: list[dict[str, Any]] = []
    for i, j, sym in pairs:
        key = (i, j) if i < j else (j, i)
        if key in seen:
            continue
        seen.add(key)
        out.append({"i": i, "j": j, "sum_off_diag_normalized": sym, "raw_count": float(cm[i, j] + cm[j, i])})
        if len(out) >= k:
            break
    return out


def per_class_f1(cm: np.ndarray) -> list[float]:
    n = cm.shape[0]
    f1: list[float] = []
    for i in range(n):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return f1


def class_centroids(features: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((n_classes, features.shape[1]), dtype=np.float64)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            out[c] = features[mask].mean(axis=0)
    return out


def normalized_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(float(np.dot(d, d))))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not AGG_CM_PATH.exists():
        raise FileNotFoundError(f"missing {AGG_CM_PATH}")
    confusions = load_aggregate_confusions()
    label_names = load_label_names()
    n_classes = max(label_names) + 1

    train_features = np.load(TRAIN_FEATURES_PATH, allow_pickle=False).astype(np.float64, copy=False)
    train_labels = np.load(TRAIN_LABELS_PATH, allow_pickle=False).astype(np.int64, copy=False)
    group_indices = resolve_feature_group_indices()
    group_names = list(group_indices)

    full_centroids = class_centroids(train_features, train_labels, n_classes)
    group_centroids = {
        gname: class_centroids(train_features[:, idx], train_labels, n_classes)
        for gname, idx in group_indices.items()
    }

    inter_pair_distances: dict[tuple[int, int], dict[str, Any]] = {}
    for i, j in combinations(range(n_classes), 2):
        per_group = {gname: normalized_euclidean(group_centroids[gname][i], group_centroids[gname][j]) for gname in group_names}
        inter_pair_distances[(i, j)] = {
            "i": i,
            "j": j,
            "i_name": label_names.get(i, str(i)),
            "j_name": label_names.get(j, str(j)),
            "full_distance": normalized_euclidean(full_centroids[i], full_centroids[j]),
            "per_group_distance": per_group,
        }

    distances_payload: dict[str, Any] = {
        "n_classes": n_classes,
        "n_pairs": len(inter_pair_distances),
        "group_names": group_names,
        "pairs": [v for v in inter_pair_distances.values()],
    }
    (OUT_DIR / "inter_genre_distances_per_group.json").write_text(json.dumps(distances_payload, indent=2), encoding="utf-8")

    top_confused: dict[str, Any] = {}
    fingerprint: dict[str, Any] = {}
    for arch_key in DEEP_KEYS:
        if arch_key not in confusions:
            continue
        cm = confusions[arch_key].astype(np.float64, copy=False)
        top = top_confused_pairs(cm, k=5)
        for entry in top:
            entry["i_name"] = label_names.get(entry["i"], str(entry["i"]))
            entry["j_name"] = label_names.get(entry["j"], str(entry["j"]))
        top_confused[arch_key] = top
        f1 = per_class_f1(cm)
        ranked = sorted(((label_names.get(i, str(i)), v) for i, v in enumerate(f1)), key=lambda t: t[1])
        row_sum = cm.sum(axis=1, keepdims=True)
        col_sum = cm.sum(axis=0, keepdims=True)
        normalized = np.where(row_sum > 0, cm / row_sum, 0.0)
        np.fill_diagonal(normalized, 0.0)
        most_confused_out_idx = int(normalized.sum(axis=1).argmax())
        most_confused_in_idx = int(normalized.sum(axis=0).argmax())
        fingerprint[arch_key] = {
            "per_class_f1_sorted_ascending": [{"class": name, "f1": float(v)} for name, v in ranked],
            "weakest_class": ranked[0][0],
            "strongest_class": ranked[-1][0],
            "most_confused_out_class": label_names.get(most_confused_out_idx, str(most_confused_out_idx)),
            "most_confused_in_class": label_names.get(most_confused_in_idx, str(most_confused_in_idx)),
        }

    (OUT_DIR / "top_confused_pairs_per_arch.json").write_text(json.dumps(top_confused, indent=2), encoding="utf-8")
    (OUT_DIR / "confusion_fingerprint_per_arch.json").write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")

    spearman_payload: dict[str, Any] = {"per_arch": {}}
    if GROUPS_CKA_PATH.exists():
        groups_cka = json.loads(GROUPS_CKA_PATH.read_text(encoding="utf-8"))
        per_arch_per_group = groups_cka.get("per_arch_per_group", {})
        for arch_key, top_pairs in top_confused.items():
            if arch_key not in per_arch_per_group:
                continue
            arch_groups = per_arch_per_group[arch_key]
            cka_vec = np.array([arch_groups.get(g, float("nan")) for g in group_names], dtype=np.float64)
            mean_distance_per_group = []
            for g in group_names:
                vals = []
                for tp in top_pairs:
                    pair_key = (tp["i"], tp["j"]) if tp["i"] < tp["j"] else (tp["j"], tp["i"])
                    pd = inter_pair_distances[pair_key]["per_group_distance"][g]
                    vals.append(pd)
                mean_distance_per_group.append(float(np.mean(vals)) if vals else float("nan"))
            mean_distance_vec = np.array(mean_distance_per_group, dtype=np.float64)
            mask = np.isfinite(cka_vec) & np.isfinite(mean_distance_vec)
            # with fewer than three valid feature groups the rank correlation is not
            # meaningful, so we emit nan rather than a spurious coefficient.
            if mask.sum() < 3:
                rho = float("nan")
                p = float("nan")
            else:
                res = spearmanr(cka_vec[mask], mean_distance_vec[mask])
                rho = float(res.correlation)
                p = float(res.pvalue)
            spearman_payload["per_arch"][arch_key] = {
                "rho": rho,
                "p_value": p,
                "n_groups": int(mask.sum()),
                "cka_per_group": {g: float(cka_vec[i]) for i, g in enumerate(group_names)},
                "mean_distance_per_group_over_top_confused": {g: mean_distance_per_group[i] for i, g in enumerate(group_names)},
            }
    else:
        spearman_payload["note"] = f"missing {GROUPS_CKA_PATH}; run scripts/04_freeze/run_canonical_cka.py first"

    (OUT_DIR / "cka_distance_spearman.json").write_text(json.dumps(spearman_payload, indent=2), encoding="utf-8")

    summary = {
        "created_at_utc": utc_now(),
        "frozen_manifest": str(MANIFEST_PATH),
        "outputs": sorted(p.name for p in OUT_DIR.iterdir()),
        "label_count": n_classes,
        "deep_archs_processed": list(top_confused),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[error-analysis] outputs:")
    for p in sorted(OUT_DIR.iterdir()):
        print(" ", p.name, p.stat().st_size)
    print("\n[error-analysis] top confused pair per arch:")
    for arch, pairs in top_confused.items():
        if pairs:
            tp = pairs[0]
            print(f"  {arch}: {tp['i_name']} <-> {tp['j_name']} (sum_norm={tp['sum_off_diag_normalized']:.3f})")
    print("\n[error-analysis] Spearman rho per arch (CKA(group) vs mean_distance(group) over top confused):")
    for arch, payload in spearman_payload.get("per_arch", {}).items():
        print(f"  {arch}: rho={payload['rho']:+.3f} p={payload['p_value']:.3f}")


if __name__ == "__main__":
    main()
