from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src.analysis.cka import (
    align_matrices_by_track_id,
    linear_cka,
    load_split_features,
    rbf_cka,
    resolve_feature_group_indices,
    validate_matrix,
)
from src.analysis.representations import (
    EXPECTED_REPRESENTATION_DIMS,
    ExtractionConfig,
    MODEL_ALIASES,
    extract_representations,
    extract_untrained_representations,
    load_representation_dir,
)


MANIFEST_PATH = REPO_ROOT / "results" / "frozen_manifest.json"
REPRESENTATIONS_DIR = REPO_ROOT / "results" / "representations"
CKA_DIR = REPO_ROOT / "results" / "cka"
DEEP_KEYS = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")
INTERMEDIATE_NAME_BY_MODEL = {
    "cnn": "block_2_mean",
    "cnn2d": "block_4_mean",
    "lstm": "sequence_mean",
    "transformer": "layer_2_time_mean",
    "mamba1": "block_2_mean",
    "mamba2": "block_2_mean",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def display_for(model_key: str) -> str:
    return model_key


def canonical_for(model_key: str) -> str:
    canonical = MODEL_ALIASES.get(model_key.lower())
    if canonical is None:
        raise ValueError(f"unknown model_key {model_key!r}")
    return canonical


def representation_dir_for(model_key: str, seed: int) -> Path:
    return REPRESENTATIONS_DIR / f"{model_key}_seed{seed}"


def ensure_representations(entries: list[dict[str, Any]]) -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for entry in entries:
        if entry.get("kind") != "deep":
            continue
        model_key = str(entry["model_key"])
        if model_key not in DEEP_KEYS:
            continue
        seed = int(entry["seed"])
        out_dir = representation_dir_for(model_key, seed)
        manifest_path = out_dir / "manifest.json"
        penultimate_path = out_dir / "penultimate.npy"
        if penultimate_path.exists() and manifest_path.exists():
            paths[(model_key, seed)] = out_dir
            continue
        run_dir = _translate_path(entry["output_dir_abs"]) if sys.platform != "win32" else Path(entry["output_dir_abs"])
        checkpoint = run_dir / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"missing best.pt: {checkpoint}")
        out_dir.mkdir(parents=True, exist_ok=True)
        config = ExtractionConfig(
            checkpoint_path=checkpoint,
            output_dir=out_dir,
            model_name=canonical_for(model_key),
            split="test",
            batch_size=32,
            num_workers=0,
            limit_samples=None,
            device=None,
            save_intermediates=True,
            source_run_dir=run_dir,
        )
        manifest = extract_representations(config)
        manifest["model_key"] = model_key
        manifest["seed"] = seed
        manifest["output_dir_abs"] = str(out_dir)
        manifest["frozen_manifest_source"] = str(MANIFEST_PATH)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        intermediate_key = INTERMEDIATE_NAME_BY_MODEL[canonical_for(model_key)]
        intermediate_path = out_dir / "intermediates" / f"{intermediate_key}.npy"
        if intermediate_path.exists():
            arr = np.load(intermediate_path)
            np.save(out_dir / "intermediate_l2.npy", arr.astype(np.float32, copy=False), allow_pickle=False)
        track_ids_path = out_dir / "track_ids.npy"
        if track_ids_path.exists():
            np.save(out_dir / "test_track_ids.npy", np.load(track_ids_path), allow_pickle=False)
        paths[(model_key, seed)] = out_dir
        print(f"  extracted {model_key}_seed{seed}")
    return paths


def load_seed42_matrices(paths: dict[tuple[str, int], Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    named: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_key in DEEP_KEYS:
        rep = load_representation_dir(paths[(model_key, 42)], array_name="penultimate")
        named[model_key] = (rep["track_ids"], rep["matrix"])
    aligned, ordered_ids = align_matrices_by_track_id(named, reference_name=DEEP_KEYS[0])
    return aligned, ordered_ids


def compute_inter_arch(matrices: dict[str, np.ndarray], method: str, seed: int = 0) -> tuple[list[str], np.ndarray]:
    names = list(matrices)
    n = len(names)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if method == "linear":
                value = linear_cka(matrices[names[i]], matrices[names[j]])
            elif method == "rbf":
                value = rbf_cka(matrices[names[i]], matrices[names[j]], sample_size=None, seed=seed)
            else:
                raise ValueError(method)
            matrix[i, j] = value
            matrix[j, i] = value
    return names, matrix


def render_heatmap(names: list[str], matrix: np.ndarray) -> str:
    header = "         " + " ".join(f"{n[:7]:>7}" for n in names)
    lines = [header]
    for i, name in enumerate(names):
        row = "  ".join(f"{matrix[i, j]:0.3f}" for j in range(len(names)))
        lines.append(f"{name[:8]:>8} {row}")
    return "\n".join(lines)


def compute_untrained_floor(track_ids: np.ndarray, trained_matrices: dict[str, np.ndarray]) -> dict[str, Any]:
    # we use randomly initialized weights on the same tracks as the floor control: any
    # cross-arch similarity here is architectural prior, not learned convergence.
    untrained: dict[str, np.ndarray] = {}
    for model_key in DEEP_KEYS:
        canonical = canonical_for(model_key)
        matrix = extract_untrained_representations(
            model_name=canonical,
            split="test",
            track_ids=track_ids,
            batch_size=32,
            num_workers=0,
            device=None,
            seed=0,
        )
        untrained[model_key] = matrix
    inter_pairs = []
    for left, right in combinations(DEEP_KEYS, 2):
        value = linear_cka(untrained[left], untrained[right])
        inter_pairs.append({"left": left, "right": right, "linear_cka": float(value)})
    trained_vs_untrained = []
    for model_key in DEEP_KEYS:
        value = linear_cka(trained_matrices[model_key], untrained[model_key])
        trained_vs_untrained.append({"model_key": model_key, "linear_cka": float(value)})
    floor = float(np.mean([row["linear_cka"] for row in inter_pairs]))
    return {
        "untrained_floor_mean_inter_arch_linear_cka": floor,
        "inter_arch_pairs": inter_pairs,
        "trained_vs_untrained_same_arch": trained_vs_untrained,
    }


def compute_within_arch_ceiling(paths: dict[tuple[str, int], Path]) -> dict[str, Any]:
    summary: dict[str, Any] = {"per_arch": {}, "all_pairs": []}
    means = []
    for model_key in DEEP_KEYS:
        seeds = sorted({seed for (mk, seed) in paths if mk == model_key})
        if len(seeds) < 2:
            continue
        named: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for seed in seeds:
            rep = load_representation_dir(paths[(model_key, seed)], array_name="penultimate")
            named[f"seed{seed}"] = (rep["track_ids"], rep["matrix"])
        aligned, _ids = align_matrices_by_track_id(named, reference_name=f"seed{seeds[0]}")
        pair_values = []
        pairs = []
        for left, right in combinations(sorted(aligned), 2):
            value = float(linear_cka(aligned[left], aligned[right]))
            pair_values.append(value)
            pair = {"model_key": model_key, "left": left, "right": right, "linear_cka": value}
            pairs.append(pair)
            summary["all_pairs"].append(pair)
        mean_val = float(np.mean(pair_values))
        summary["per_arch"][model_key] = {
            "n_seeds": len(seeds),
            "n_pairs": len(pair_values),
            "mean_linear_cka": mean_val,
            "pairs": pairs,
        }
        means.append(mean_val)
    summary["mean_across_archs"] = float(np.mean(means)) if means else None
    return summary


def compute_classical_alignments(matrices: dict[str, np.ndarray], track_ids: np.ndarray) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, list[str]]:
    feature_matrix, _labels, feature_track_ids = load_split_features("test")
    aligned, ordered_ids = align_matrices_by_track_id(
        {"classical": (feature_track_ids, feature_matrix), "ref": (track_ids, matrices[DEEP_KEYS[0]])},
        reference_name="ref",
    )
    classical = aligned["classical"]
    if not np.array_equal(ordered_ids, track_ids):
        # alignment did not preserve our reference order, so we reindex the classical
        # rows to match the deep matrices track for track before computing CKA.
        reorder_map = {int(t): i for i, t in enumerate(feature_track_ids.tolist())}
        idx = np.asarray([reorder_map[int(t)] for t in track_ids.tolist()], dtype=np.int64)
        classical = feature_matrix[idx]
    full = {}
    for model_key, matrix in matrices.items():
        full[model_key] = float(linear_cka(matrix, classical))
    group_indices = resolve_feature_group_indices()
    group_names = list(group_indices)
    group_matrix = np.zeros((len(matrices), len(group_names)), dtype=np.float64)
    groups_payload: dict[str, dict[str, float]] = {}
    for i, model_key in enumerate(matrices):
        groups_payload[model_key] = {}
        for j, group_name in enumerate(group_names):
            value = float(linear_cka(matrices[model_key], classical[:, group_indices[group_name]]))
            group_matrix[i, j] = value
            groups_payload[model_key][group_name] = value
    full_payload = {
        "per_arch_full_linear_cka": full,
        "feature_matrix_shape": list(classical.shape),
        "n_samples": int(classical.shape[0]),
    }
    groups_out = {
        "model_keys": list(matrices),
        "group_names": group_names,
        "matrix": group_matrix.tolist(),
        "per_arch_per_group": groups_payload,
    }
    return full_payload, groups_out, classical, group_names


def compute_feature_collinearity(classical: np.ndarray) -> dict[str, Any]:
    group_indices = resolve_feature_group_indices()
    names = list(group_indices)
    abs_pearsons: list[float] = []
    pair_values: list[dict[str, Any]] = []
    for left, right in combinations(names, 2):
        a = classical[:, group_indices[left]]
        b = classical[:, group_indices[right]]
        a_flat = a - a.mean(axis=0, keepdims=True)
        b_flat = b - b.mean(axis=0, keepdims=True)
        rs = []
        for col_a in range(a_flat.shape[1]):
            for col_b in range(b_flat.shape[1]):
                num = float(np.sum(a_flat[:, col_a] * b_flat[:, col_b]))
                den = float(np.sqrt(np.sum(a_flat[:, col_a] ** 2) * np.sum(b_flat[:, col_b] ** 2)))
                if den > 0.0:
                    rs.append(abs(num / den))
        mean_abs_r = float(np.mean(rs)) if rs else float("nan")
        abs_pearsons.append(mean_abs_r)
        pair_values.append({"left": left, "right": right, "mean_abs_pearson_r": mean_abs_r})
    return {
        "n_pairs": len(pair_values),
        "mean_abs_pearson_r_across_pairs": float(np.mean(abs_pearsons)) if abs_pearsons else None,
        "pairs": pair_values,
    }


def main() -> None:
    REPRESENTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    CKA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    print("[cka] ensuring representations for all deep entries (5 archs x 5 seeds)")
    paths = ensure_representations(entries)

    print("[cka] loading seed-42 matrices for inter-arch CKA")
    seed42_matrices, seed42_track_ids = load_seed42_matrices(paths)
    np.save(CKA_DIR / "seed42_track_ids.npy", seed42_track_ids, allow_pickle=False)

    names_lin, inter_lin = compute_inter_arch(seed42_matrices, method="linear")
    np.save(CKA_DIR / "inter_arch_linear.npy", inter_lin, allow_pickle=False)
    (CKA_DIR / "inter_arch_linear_labels.json").write_text(json.dumps(names_lin, indent=2), encoding="utf-8")

    if not np.allclose(inter_lin, inter_lin.T, atol=1e-8):
        raise AssertionError("inter_arch_linear is not symmetric")
    if not np.allclose(np.diag(inter_lin), 1.0, atol=1e-6):
        raise AssertionError("inter_arch_linear diagonal != 1")
    off_diag = inter_lin[~np.eye(len(names_lin), dtype=bool)]
    if (off_diag < 0).any() or (off_diag > 1).any():
        raise AssertionError(f"inter_arch_linear has out-of-range off-diagonals: min={off_diag.min()} max={off_diag.max()}")

    print("[cka] computing inter-arch RBF CKA (median heuristic, full sample)")
    names_rbf, inter_rbf = compute_inter_arch(seed42_matrices, method="rbf")
    np.save(CKA_DIR / "inter_arch_rbf.npy", inter_rbf, allow_pickle=False)
    (CKA_DIR / "inter_arch_rbf_labels.json").write_text(json.dumps(names_rbf, indent=2), encoding="utf-8")

    print("[cka] computing untrained floor")
    floor_payload = compute_untrained_floor(seed42_track_ids, seed42_matrices)
    (CKA_DIR / "untrained_floor.json").write_text(json.dumps(floor_payload, indent=2), encoding="utf-8")

    print("[cka] computing within-arch ceiling across 5 seeds")
    ceiling_payload = compute_within_arch_ceiling(paths)
    (CKA_DIR / "within_arch_ceiling.json").write_text(json.dumps(ceiling_payload, indent=2), encoding="utf-8")

    print("[cka] computing arch-to-classical (full + groups)")
    full_payload, groups_payload, classical, group_names = compute_classical_alignments(seed42_matrices, seed42_track_ids)
    (CKA_DIR / "arch_to_classical_full.json").write_text(json.dumps(full_payload, indent=2), encoding="utf-8")
    (CKA_DIR / "arch_to_classical_groups.json").write_text(json.dumps(groups_payload, indent=2), encoding="utf-8")

    print("[cka] computing novelty gap")
    inter_mean_per_arch: dict[str, float] = {}
    n_models = len(names_lin)
    for i, name in enumerate(names_lin):
        offdiag = [inter_lin[i, j] for j in range(n_models) if j != i]
        inter_mean_per_arch[name] = float(np.mean(offdiag))
    novelty_payload = {}
    for name in names_lin:
        # novelty gap is how much closer an arch sits to its peers than to the classical
        # features: a large positive gap is the deep-vs-classical signal we report.
        gap = inter_mean_per_arch[name] - full_payload["per_arch_full_linear_cka"][name]
        novelty_payload[name] = {
            "mean_inter_arch_linear_cka": inter_mean_per_arch[name],
            "linear_cka_to_classical_full": full_payload["per_arch_full_linear_cka"][name],
            "novelty_gap": float(gap),
        }
    (CKA_DIR / "novelty_gap.json").write_text(json.dumps(novelty_payload, indent=2), encoding="utf-8")

    print("[cka] computing classical feature collinearity")
    collin_payload = compute_feature_collinearity(classical)
    (CKA_DIR / "feature_collinearity.json").write_text(json.dumps(collin_payload, indent=2), encoding="utf-8")

    floor_mean = floor_payload["untrained_floor_mean_inter_arch_linear_cka"]
    inter_offdiag_mean = float(np.mean(inter_lin[~np.eye(n_models, dtype=bool)]))
    if floor_mean >= inter_offdiag_mean:
        print(f"  WARN: untrained floor {floor_mean:.3f} not strictly below trained inter-arch mean {inter_offdiag_mean:.3f}")
    for model_key, payload in ceiling_payload["per_arch"].items():
        if payload["mean_linear_cka"] <= inter_mean_per_arch[model_key]:
            print(f"  WARN: ceiling {payload['mean_linear_cka']:.3f} for {model_key} not above its inter-arch mean {inter_mean_per_arch[model_key]:.3f}")

    print("\n=== inter_arch_linear (5x5) ===")
    print(render_heatmap(names_lin, inter_lin))
    print(f"\nuntrained_floor_mean_inter_arch_linear_cka = {floor_mean:.4f}")
    print(f"trained_inter_arch_offdiag_mean = {inter_offdiag_mean:.4f}")
    print("within_arch_ceiling per arch (mean linear CKA):")
    for k, v in ceiling_payload["per_arch"].items():
        print(f"  {k}: {v['mean_linear_cka']:.4f} ({v['n_pairs']} pairs)")
    print("novelty_gap:")
    for k, v in novelty_payload.items():
        print(f"  {k}: {v['novelty_gap']:+.4f}")
    summary = {
        "created_at_utc": utc_now(),
        "frozen_manifest": str(MANIFEST_PATH),
        "model_keys": list(names_lin),
        "n_test_samples": int(seed42_track_ids.shape[0]),
        "trained_inter_arch_offdiag_mean_linear_cka": inter_offdiag_mean,
        "untrained_floor_mean_inter_arch_linear_cka": floor_mean,
        "ceiling_mean_across_archs_linear_cka": ceiling_payload.get("mean_across_archs"),
        "outputs": sorted(p.name for p in CKA_DIR.iterdir()),
    }
    (CKA_DIR / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
