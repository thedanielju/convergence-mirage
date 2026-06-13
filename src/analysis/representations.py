from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FMASpectrogramDataset
from src.training.deep import MODEL_DEFAULTS, build_model, count_parameters


REPO_ROOT = Path(__file__).resolve().parents[2]
# the penultimate width each architecture exposes. we assert against these so a
# silently reshaped representation cannot slip into the cka comparison.
EXPECTED_REPRESENTATION_DIMS = {
    "cnn": 512,
    "cnn2d": 512,
    "lstm": 512,
    "transformer": 256,
    "mamba1": 256,
    "mamba2": 256,
}

MODEL_ALIASES = {
    "cnn1d": "cnn",
    "cnn": "cnn",
    "cnn2d": "cnn2d",
    "cnn2d_final_baseline": "cnn2d",
    "bilstm": "lstm",
    "bilstm_attention": "lstm",
    "lstm": "lstm",
    "transformer": "transformer",
    "transformer_classifier": "transformer",
    "mamba1": "mamba1",
    "mamba1_classifier": "mamba1",
    "mamba2": "mamba2",
    "mamba2_classifier": "mamba2",
}


@dataclass(frozen=True)
class ExtractionConfig:
    checkpoint_path: Path
    output_dir: Path
    model_name: str | None = None
    split: str = "test"
    batch_size: int = 32
    num_workers: int = 0
    limit_samples: int | None = None
    device: str | None = None
    save_intermediates: bool = True
    source_run_dir: Path | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_model_name(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(key)


def infer_model_name(checkpoint_path: Path, run_dir: Path | None = None, explicit: str | None = None) -> str:
    model_name = normalize_model_name(explicit)
    if model_name is not None:
        return model_name

    search_dirs: list[Path] = []
    if run_dir is not None:
        search_dirs.append(run_dir)
    search_dirs.append(checkpoint_path.parent)

    for directory in search_dirs:
        for filename in ("config.json", "run_manifest.json", "metrics.json"):
            payload = _read_json(directory / filename)
            if not payload:
                continue
            candidates = [
                payload.get("model"),
                payload.get("display_name"),
                payload.get("model_name"),
            ]
            config = payload.get("config")
            if isinstance(config, dict):
                candidates.extend([config.get("model"), config.get("display_name")])
            for candidate in candidates:
                model_name = normalize_model_name(candidate)
                if model_name is not None:
                    return model_name

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        checkpoint = None
    if isinstance(checkpoint, dict):
        for candidate in (checkpoint.get("model"), checkpoint.get("model_name")):
            model_name = normalize_model_name(candidate)
            if model_name is not None:
                return model_name
        config = checkpoint.get("config")
        if isinstance(config, dict):
            for candidate in (config.get("model"), config.get("display_name")):
                model_name = normalize_model_name(candidate)
                if model_name is not None:
                    return model_name

    path_hint = str(checkpoint_path).lower()
    for alias, canonical in sorted(MODEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in path_hint:
            return canonical
    raise ValueError("could not infer model name; pass --model explicitly")


def _state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("checkpoint does not contain a model state dict")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_model(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
    strict: bool = True,
) -> torch.nn.Module:
    if model_name not in MODEL_DEFAULTS:
        raise ValueError(f"unsupported model for representation extraction: {model_name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _strip_module_prefix(_state_dict_from_checkpoint(checkpoint))
    dropout = MODEL_DEFAULTS[model_name].dropout
    model = build_model(model_name, dropout=dropout)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def limited_dataset(split: str, limit: int | None) -> torch.utils.data.Dataset:
    dataset = FMASpectrogramDataset(split=split, apply_specaugment=False)
    if limit is None:
        return dataset
    count = min(int(limit), len(dataset))
    return Subset(dataset, list(range(count)))


def dataset_records(dataset: torch.utils.data.Dataset) -> list[dict[str, Any]]:
    if isinstance(dataset, Subset):
        base_records = dataset_records(dataset.dataset)
        return [base_records[int(index)] for index in dataset.indices]
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("dataset does not expose ordered records")
    return list(records)


def dataset_track_ids_and_labels(dataset: torch.utils.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    label_to_id = getattr(base, "label_to_id", None)
    if label_to_id is None:
        raise ValueError("dataset does not expose label_to_id")
    records = dataset_records(dataset)
    track_ids = np.asarray([int(record["track_id"]) for record in records], dtype=np.int64)
    labels = np.asarray([int(label_to_id[str(record["genre_top"])]) for record in records], dtype=np.int64)
    return track_ids, labels


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def validate_extracted_arrays(
    penultimate: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    track_ids: np.ndarray,
    expected_dim: int,
) -> None:
    n = int(track_ids.shape[0])
    shapes = {
        "penultimate": penultimate.shape[0],
        "logits": logits.shape[0],
        "probabilities": probabilities.shape[0],
        "predictions": predictions.shape[0],
        "labels": labels.shape[0],
    }
    bad = {name: count for name, count in shapes.items() if int(count) != n}
    if bad:
        raise ValueError(f"sample alignment mismatch against track_ids length {n}: {bad}")
    if penultimate.ndim != 2 or penultimate.shape[1] != expected_dim:
        raise ValueError(f"expected penultimate shape (N, {expected_dim}), found {penultimate.shape}")
    for name, array in {
        "penultimate": penultimate,
        "logits": logits,
        "probabilities": probabilities,
    }.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if np.unique(track_ids).shape[0] != track_ids.shape[0]:
        raise ValueError("track_ids contain duplicates")


def extract_arrays(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collect_intermediates: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    penultimate_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    intermediate_batches: dict[str, list[np.ndarray]] = {}

    with torch.no_grad():
        for inputs, _targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(inputs)
            # we pull the pre-classifier features through the model's own
            # get_penultimate hook so every architecture exposes one comparable
            # vector regardless of its internal pooling.
            penultimate = model.get_penultimate(inputs)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            penultimate_batches.append(_to_numpy(penultimate))
            logits_batches.append(_to_numpy(logits))
            probability_batches.append(_to_numpy(probabilities))
            prediction_batches.append(predictions.detach().cpu().numpy().astype(np.int64, copy=False))

            if collect_intermediates and hasattr(model, "get_intermediate_representations"):
                batch_intermediates = model.get_intermediate_representations(inputs)
                for name, tensor in batch_intermediates.items():
                    if tensor.ndim <= 2:
                        value = tensor
                    else:
                        value = tensor.flatten(1)
                    intermediate_batches.setdefault(name, []).append(_to_numpy(value))

    penultimate_array = np.concatenate(penultimate_batches, axis=0).astype(np.float32, copy=False)
    logits_array = np.concatenate(logits_batches, axis=0).astype(np.float32, copy=False)
    probabilities_array = np.concatenate(probability_batches, axis=0).astype(np.float32, copy=False)
    predictions_array = np.concatenate(prediction_batches, axis=0).astype(np.int64, copy=False)
    intermediates = {
        name: np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        for name, batches in intermediate_batches.items()
    }
    return penultimate_array, intermediates, logits_array, probabilities_array, predictions_array


def extract_representations(config: ExtractionConfig) -> dict[str, Any]:
    checkpoint_path = config.checkpoint_path.resolve()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = infer_model_name(checkpoint_path, run_dir=config.source_run_dir, explicit=config.model_name)
    expected_dim = EXPECTED_REPRESENTATION_DIMS[model_name]
    device = torch.device(config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    model = load_checkpoint_model(checkpoint_path, model_name=model_name, device=device, strict=True)
    dataset = limited_dataset(config.split, config.limit_samples)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    track_ids, labels = dataset_track_ids_and_labels(dataset)

    started = time.perf_counter()
    penultimate, intermediates, logits, probabilities, predictions = extract_arrays(
        model=model,
        loader=loader,
        device=device,
        collect_intermediates=config.save_intermediates,
    )
    elapsed = time.perf_counter() - started
    validate_extracted_arrays(penultimate, logits, probabilities, predictions, labels, track_ids, expected_dim)

    np.save(output_dir / "penultimate.npy", penultimate, allow_pickle=False)
    np.save(output_dir / "labels.npy", labels, allow_pickle=False)
    np.save(output_dir / "track_ids.npy", track_ids, allow_pickle=False)
    np.save(output_dir / "logits.npy", logits, allow_pickle=False)
    np.save(output_dir / "probabilities.npy", probabilities, allow_pickle=False)
    np.save(output_dir / "predictions.npy", predictions, allow_pickle=False)

    intermediate_shapes: dict[str, list[int]] = {}
    if config.save_intermediates:
        intermediate_dir = output_dir / "intermediates"
        intermediate_dir.mkdir(exist_ok=True)
        for name, array in intermediates.items():
            if not np.isfinite(array).all():
                raise ValueError(f"intermediate {name} contains NaN or Inf")
            np.save(intermediate_dir / f"{name}.npy", array, allow_pickle=False)
            intermediate_shapes[name] = list(array.shape)

    manifest = {
        "created_at_utc": utc_now(),
        "entrypoint": "scripts/03_analysis/extract_representations.py",
        "command": " ".join(sys.argv),
        "source_run_dir": str(config.source_run_dir.resolve()) if config.source_run_dir else None,
        "checkpoint_path": str(checkpoint_path),
        "model": model_name,
        "expected_representation_dim": expected_dim,
        "parameter_count": count_parameters(model),
        "split": config.split,
        "sample_count": int(track_ids.shape[0]),
        "limit_samples": config.limit_samples,
        "device": str(device),
        "specaugment": "disabled",
        "eval_mode": not model.training,
        "arrays": {
            "penultimate": list(penultimate.shape),
            "labels": list(labels.shape),
            "track_ids": list(track_ids.shape),
            "logits": list(logits.shape),
            "probabilities": list(probabilities.shape),
            "predictions": list(predictions.shape),
            "intermediates": intermediate_shapes,
        },
        "runtime_seconds": float(elapsed),
        "finite_checks": {
            "penultimate": bool(np.isfinite(penultimate).all()),
            "logits": bool(np.isfinite(logits).all()),
            "probabilities": bool(np.isfinite(probabilities).all()),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_representation_dir(path: Path, array_name: str = "penultimate") -> dict[str, Any]:
    path = path.resolve()
    matrix = np.load(path / f"{array_name}.npy")
    track_ids = np.load(path / "track_ids.npy")
    labels = np.load(path / "labels.npy")
    manifest = _read_json(path / "manifest.json") or {}
    if matrix.ndim != 2:
        raise ValueError(f"{path}/{array_name}.npy must be 2D, found {matrix.shape}")
    if matrix.shape[0] != track_ids.shape[0] or labels.shape[0] != track_ids.shape[0]:
        raise ValueError(f"representation directory is not sample aligned: {path}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"representation matrix contains NaN or Inf: {path}")
    return {
        "path": path,
        "matrix": matrix,
        "track_ids": track_ids.astype(np.int64),
        "labels": labels.astype(np.int64),
        "manifest": manifest,
    }


# we run a freshly initialized model over the exact same track_ids to get an
# untrained-floor representation. the cka analysis subtracts this floor so we do
# not credit random architecture structure as learned convergence.
def extract_untrained_representations(
    model_name: str,
    split: str,
    track_ids: np.ndarray,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    set_seed(seed)
    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    base_dataset = FMASpectrogramDataset(split=split, apply_specaugment=False)
    all_ids = np.asarray([int(record["track_id"]) for record in base_dataset.records], dtype=np.int64)
    positions = {int(track_id): index for index, track_id in enumerate(all_ids.tolist())}
    missing = [int(track_id) for track_id in track_ids.tolist() if int(track_id) not in positions]
    if missing:
        raise ValueError(f"cannot extract untrained floor; {len(missing)} track_ids are absent from split {split}")
    subset = Subset(base_dataset, [positions[int(track_id)] for track_id in track_ids.tolist()])
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch_device.type == "cuda",
    )
    model = build_model(model_name, dropout=MODEL_DEFAULTS[model_name].dropout).to(torch_device)
    model.eval()
    penultimate, _intermediates, _logits, _probabilities, _predictions = extract_arrays(
        model=model,
        loader=loader,
        device=torch_device,
        collect_intermediates=False,
    )
    expected_dim = EXPECTED_REPRESENTATION_DIMS[model_name]
    if penultimate.shape != (track_ids.shape[0], expected_dim):
        raise ValueError(f"unexpected untrained representation shape {penultimate.shape}")
    return penultimate
