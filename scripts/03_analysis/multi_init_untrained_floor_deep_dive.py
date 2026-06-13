from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ARCHES = ("cnn2d", "bilstm", "transformer", "mamba1", "mamba2")


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").exists() and (candidate / "scripts").exists():
            return candidate
        if (candidate / "code" / "music_classification_code" / "src").exists():
            return candidate / "code" / "music_classification_code"
    raise RuntimeError(f"could not locate music_classification_code root from {start}")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FMASpectrogramDataset
from src.models.cnn import CNN2D
from src.models.lstm import BiLSTMClassifier
from src.models.mamba import Mamba1Classifier, Mamba2Classifier
from src.models.transformer import TransformerClassifier


MODEL_DROPOUT = {
    "cnn2d": 0.0,
    "lstm": 0.3,
    "transformer": 0.1,
    "mamba1": 0.1,
    "mamba2": 0.1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_matrix(name: str, matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, found {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def center_gram(gram: np.ndarray) -> np.ndarray:
    gram = np.asarray(gram, dtype=np.float64)
    return gram - gram.mean(axis=0, keepdims=True) - gram.mean(axis=1, keepdims=True) + gram.mean()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = validate_matrix("x", x)
    y = validate_matrix("y", y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"sample mismatch: x has {x.shape[0]}, y has {y.shape[0]}")
    k = center_gram(x @ x.T)
    l = center_gram(y @ y.T)
    denominator = float(np.sqrt(np.sum(k * k) * np.sum(l * l)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.clip(float(np.sum(k * l)) / denominator, -1.0, 1.0))


def align_matrices_by_track_id(
    matrices: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    maps: dict[str, dict[int, int]] = {}
    for name, (track_ids, _matrix) in matrices.items():
        ids = np.asarray(track_ids, dtype=np.int64)
        if np.unique(ids).shape[0] != ids.shape[0]:
            raise ValueError(f"{name} track_ids contain duplicates")
        maps[name] = {int(track_id): int(index) for index, track_id in enumerate(ids.tolist())}
    common_ids = set.intersection(*(set(track_map) for track_map in maps.values()))
    reference_ids = np.asarray(matrices[reference_name][0], dtype=np.int64)
    ordered_ids = np.asarray([int(track_id) for track_id in reference_ids.tolist() if int(track_id) in common_ids], dtype=np.int64)
    aligned: dict[str, np.ndarray] = {}
    for name, (_track_ids, matrix) in matrices.items():
        indices = [maps[name][int(track_id)] for track_id in ordered_ids.tolist()]
        aligned[name] = validate_matrix(name, matrix)[indices]
    return aligned, ordered_ids


def save_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[float(value) for value in row]])


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "cnn2d":
        return CNN2D()
    if model_name == "lstm":
        return BiLSTMClassifier(dropout=MODEL_DROPOUT[model_name])
    if model_name == "transformer":
        return TransformerClassifier(dropout=MODEL_DROPOUT[model_name])
    if model_name == "mamba1":
        return Mamba1Classifier(dropout=MODEL_DROPOUT[model_name])
    if model_name == "mamba2":
        return Mamba2Classifier(dropout=MODEL_DROPOUT[model_name])
    raise ValueError(f"unsupported model: {model_name}")


def load_representation_dir(path: Path) -> dict[str, Any]:
    path = path.resolve()
    matrix = np.load(path / "penultimate.npy")
    track_ids = np.load(path / "track_ids.npy").astype(np.int64)
    labels = np.load(path / "labels.npy").astype(np.int64)
    if matrix.ndim != 2:
        raise ValueError(f"{path}/penultimate.npy must be 2D, found {matrix.shape}")
    if matrix.shape[0] != track_ids.shape[0] or labels.shape[0] != track_ids.shape[0]:
        raise ValueError(f"{path}: representation arrays are not aligned")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: representation matrix contains non-finite values")
    return {"path": path, "matrix": matrix, "track_ids": track_ids, "labels": labels}


def extract_untrained_representations(
    model_name: str,
    split: str,
    track_ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: str,
    seed: int,
) -> np.ndarray:
    # we forward-pass a randomly initialized model without ever training it, so the
    # penultimate features reflect architecture and init alone, not anything learned.
    set_global_seed(seed)
    torch_device = torch.device(device)
    dataset = FMASpectrogramDataset(split=split, apply_specaugment=False)
    all_ids = np.asarray([int(record["track_id"]) for record in dataset.records], dtype=np.int64)
    positions = {int(track_id): index for index, track_id in enumerate(all_ids.tolist())}
    missing = [int(track_id) for track_id in track_ids.tolist() if int(track_id) not in positions]
    if missing:
        raise ValueError(f"{len(missing)} requested track_ids are absent from split {split}")
    subset = Subset(dataset, [positions[int(track_id)] for track_id in track_ids.tolist()])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    model = build_model(model_name).to(torch_device)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device=torch_device, dtype=torch.float32, non_blocking=True)
            batches.append(model.get_penultimate(inputs).detach().float().cpu().numpy())
    matrix = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if matrix.shape[0] != track_ids.shape[0]:
        raise ValueError(f"{model_name}: expected {track_ids.shape[0]} rows, got {matrix.shape[0]}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{model_name}: untrained representation contains non-finite values")
    return matrix


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hub_root_from_repo(repo_root: Path) -> Path:
    if (repo_root / "results" / "representations").exists():
        return repo_root
    candidate = repo_root.parents[1]
    if (candidate / "results" / "representations").exists():
        return candidate
    return repo_root


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def matrix_rows(labels: list[str], matrix: np.ndarray, init_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if i < j:
                rows.append({"init_seed": init_seed, "left": left, "right": right, "linear_cka": float(matrix[i, j])})
    return rows


def summarize_pair_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([float(row["linear_cka"]) for row in rows], dtype=np.float64)
    by_pair: dict[str, list[float]] = {}
    for row in rows:
        key = f"{row['left']}__{row['right']}"
        by_pair.setdefault(key, []).append(float(row["linear_cka"]))
    return {
        "mean_inter_arch_linear_cka": float(np.mean(values)),
        "std_inter_arch_linear_cka": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min_inter_arch_linear_cka": float(np.min(values)),
        "max_inter_arch_linear_cka": float(np.max(values)),
        "n_values": int(values.size),
        "per_pair": {
            key: {
                "mean": float(np.mean(pair_values)),
                "std": float(np.std(pair_values, ddof=1)) if len(pair_values) > 1 else 0.0,
                "min": float(np.min(pair_values)),
                "max": float(np.max(pair_values)),
                "n": len(pair_values),
            }
            for key, pair_values in sorted(by_pair.items())
        },
    }


def write_report(path: Path, summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    gap = summary.get("trained_minus_multi_init_floor_mean")
    lines = [
        "# Multi-Init Untrained-Floor Deep Dive",
        "",
        f"Created: {manifest['created_at_utc']}",
        "",
        "Protocol: randomly initialized versions of the selected deep architectures were forward-passed over the aligned seed-42 test track order. For each initialization seed, the script computed the inter-architecture linear CKA matrix and trained-vs-untrained same-architecture CKA against the local seed-42 trained representations.",
        "",
        "## Headline",
        "",
        f"- Init seeds: {', '.join(str(seed) for seed in manifest['init_seeds'])}",
        f"- Architectures: {', '.join(manifest['arches'])}",
        f"- Multi-init inter-arch floor mean: {summary['mean_inter_arch_linear_cka']:.4f}",
        f"- Multi-init inter-arch floor std across pair/init values: {summary['std_inter_arch_linear_cka']:.4f}",
        f"- Original trained inter-arch mean: {summary['trained_inter_arch_offdiag_mean_linear_cka']:.4f}",
        f"- Trained minus multi-init floor mean: {gap:.4f}" if gap is not None else "- Trained minus multi-init floor mean: unavailable",
        "",
        "## Output Files",
        "",
        "- `inter_arch_by_init.csv`: one row per init seed and architecture pair.",
        "- `trained_vs_untrained_same_arch.csv`: same-architecture trained-vs-random CKA for each init seed.",
        "- `inter_arch_matrix_seed*.csv`: matrix form for each init seed.",
        "- `summary.json`: machine-readable aggregate summary.",
        "- `manifest.json`: command, parameters, input paths, and runtime environment.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-init untrained CKA floor deep dive.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--hub-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["training", "validation", "test"])
    parser.add_argument("--arch", nargs="+", default=list(ARCHES), choices=list(ARCHES))
    parser.add_argument("--init-seed", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--trained-mean", type=float, default=0.6853084147164366)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    hub_root = (args.hub_root.resolve() if args.hub_root else hub_root_from_repo(repo_root))
    output_dir = (args.output_dir or (hub_root / "results" / "cka_untrained_floor_deep_dive")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arches = tuple(args.arch)
    rep_dirs = [hub_root / "results" / "representations" / f"{arch}_seed42" for arch in arches]
    reps = {arch: load_representation_dir(path) for arch, path in zip(arches, rep_dirs)}
    matrices_for_alignment = {arch: (payload["track_ids"], payload["matrix"]) for arch, payload in reps.items()}
    trained_aligned, aligned_track_ids = align_matrices_by_track_id(matrices_for_alignment, reference_name=arches[0])

    inter_rows: list[dict[str, Any]] = []
    same_arch_rows: list[dict[str, Any]] = []
    runtime_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # we repeat the floor measurement across several init seeds so the untrained
    # baseline is an average over initializations rather than one lucky draw.
    for init_seed in args.init_seed:
        set_global_seed(init_seed)
        untrained: dict[str, np.ndarray] = {}
        for arch in arches:
            model_name = "lstm" if arch == "bilstm" else arch
            untrained[arch] = extract_untrained_representations(
                model_name=model_name,
                split=args.split,
                track_ids=aligned_track_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=runtime_device,
                seed=init_seed,
            )
            same_arch_rows.append(
                {
                    "init_seed": init_seed,
                    "model_key": arch,
                    "trained_vs_untrained_linear_cka": float(linear_cka(trained_aligned[arch], untrained[arch])),
                }
            )

        # the off-diagonal cka between untrained architectures is the floor, the
        # cross-architecture similarity that exists before any learning happens.
        labels = list(arches)
        matrix = np.eye(len(labels), dtype=np.float64)
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if i < j:
                    value = linear_cka(untrained[left], untrained[right])
                    matrix[i, j] = value
                    matrix[j, i] = value
        save_matrix_csv(output_dir / f"inter_arch_matrix_seed{init_seed}.csv", labels, matrix)
        inter_rows.extend(matrix_rows(labels, matrix, init_seed=init_seed))

    summary = summarize_pair_rows(inter_rows)
    summary["trained_inter_arch_offdiag_mean_linear_cka"] = float(args.trained_mean)
    summary["trained_minus_multi_init_floor_mean"] = float(args.trained_mean - summary["mean_inter_arch_linear_cka"])
    same_by_arch: dict[str, list[float]] = {}
    for row in same_arch_rows:
        same_by_arch.setdefault(str(row["model_key"]), []).append(float(row["trained_vs_untrained_linear_cka"]))
    summary["trained_vs_untrained_same_arch"] = {
        arch: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": len(values),
        }
        for arch, values in sorted(same_by_arch.items())
    }

    manifest = {
        "created_at_utc": utc_now(),
        "command": " ".join(sys.argv),
        "repo_root": str(repo_root),
        "hub_root": str(hub_root),
        "split": args.split,
        "sample_count": int(aligned_track_ids.shape[0]),
        "arches": list(arches),
        "init_seeds": [int(seed) for seed in args.init_seed],
        "device": runtime_device,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "input_representation_dirs": [str(path.resolve()) for path in rep_dirs],
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.name.startswith("inter_arch_matrix_seed"))
        + ["aligned_track_ids.npy", "inter_arch_by_init.csv", "manifest.json", "report.md", "summary.json", "trained_vs_untrained_same_arch.csv"],
        "notes": [
            "Uses local seed-42 trained representations for same-architecture trained-vs-untrained comparison.",
            "The inter-architecture floor is computed from random untrained forward passes on the aligned test track order.",
        ],
    }

    np.save(output_dir / "aligned_track_ids.npy", aligned_track_ids, allow_pickle=False)
    write_csv(output_dir / "inter_arch_by_init.csv", inter_rows)
    write_csv(output_dir / "trained_vs_untrained_same_arch.csv", same_arch_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(output_dir / "report.md", summary, manifest)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
