from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARTIFACT_SCHEMA_VERSION = 2


# we keep these high-volume fields out of per-epoch csv/jsonl logs; they are
# useful while computing metrics but bloat the row-level history.
LARGE_METRIC_KEYS = {
    "targets",
    "predictions",
    "probabilities",
    "classification_report",
    "classification_report_text",
    "confusion_matrix",
    "confusion_matrix_normalized",
    "top_confusions",
}


# convert tensors, numpy arrays, numpy scalars, and non-finite floats into json
# friendly values. this keeps all run artifacts readable without custom tools.
def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value



def prepare_output_dir(output_dir: Path, overwrite: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def append_history_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_row = {key: to_jsonable(value) for key, value in row.items()}
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_row)


def append_text_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# keep per-epoch rows compact and stable. full reports/confusion matrices are
# saved separately after final evaluation.
def flatten_epoch_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key not in LARGE_METRIC_KEYS and (isinstance(value, (int, float, str, bool)) or value is None)
    }


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def get_grad_scaler_scale(scaler: torch.amp.GradScaler | None) -> float | None:
    if scaler is None or not scaler.is_enabled():
        return None
    return float(scaler.get_scale())


def collect_cuda_stats(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "cuda_memory_allocated_mb": None,
            "cuda_memory_reserved_mb": None,
            "cuda_peak_memory_allocated_mb": None,
            "cuda_peak_memory_reserved_mb": None,
        }

    return {
        "cuda_memory_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
        "cuda_memory_reserved_mb": float(torch.cuda.memory_reserved() / (1024**2)),
        "cuda_peak_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        "cuda_peak_memory_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
    }


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    cuda_memory = collect_cuda_stats(torch.device("cuda")) if cuda_available else collect_cuda_stats(torch.device("cpu"))
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pytorch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_memory": cuda_memory,
        "backend_flags": {
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }


def format_duration(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(max(0.0, seconds)))


# produce a readable multi-line epoch log. we also write the same row to
# machine-readable csv/jsonl, so here we prioritize live monitoring clarity.
def format_epoch_log(row: dict[str, Any]) -> str:
    epoch = int(row["epoch"])
    max_epochs = int(row["max_epochs"])
    patience_used = int(row["patience_used"])
    patience = int(row["patience"])
    lines = [
        f"epoch={epoch:03d}/{max_epochs:03d} lr={row['learning_rate']:.8f} epoch_time={format_duration(float(row['epoch_seconds']))}",
        (
            f"train loss={row['train_loss']:.6f} acc={row['train_accuracy']:.6f} "
            f"macro_f1={row['train_macro_f1']:.6f} weighted_f1={row['train_weighted_f1']:.6f} "
            f"balanced_acc={row['train_balanced_accuracy']:.6f}"
        ),
        (
            f"val   loss={row['val_loss']:.6f} acc={row['val_accuracy']:.6f} "
            f"macro_f1={row['val_macro_f1']:.6f} weighted_f1={row['val_weighted_f1']:.6f} "
            f"balanced_acc={row['val_balanced_accuracy']:.6f}"
        ),
        (
            f"best_val_macro_f1={row['best_val_macro_f1']:.6f} best_epoch={row['best_epoch']} "
            f"patience={patience_used}/{patience} samples_per_sec={row['samples_per_sec']:.2f}"
        ),
    ]

    if row.get("cuda_peak_memory_allocated_mb") is not None:
        lines.append(
            f"cuda peak_allocated={row['cuda_peak_memory_allocated_mb']:.1f}mb "
            f"peak_reserved={row['cuda_peak_memory_reserved_mb']:.1f}mb"
        )
    if row.get("train_grad_norm_max") is not None:
        lines.append(
            f"grad_norm mean={row['train_grad_norm_mean']:.4f} "
            f"max={row['train_grad_norm_max']:.4f} last={row['train_grad_norm_last']:.4f}"
        )

    return "\n".join(lines)


def save_evaluation_artifacts(output_dir: Path, evaluation: dict[str, Any]) -> None:
    np.save(output_dir / "confusion_matrix.npy", evaluation["confusion_matrix"], allow_pickle=False)
    np.save(output_dir / "confusion_matrix_normalized.npy", evaluation["confusion_matrix_normalized"], allow_pickle=False)
    (output_dir / "report.txt").write_text(str(evaluation["classification_report_text"]), encoding="utf-8")
    write_json(output_dir / "classification_report.json", evaluation["classification_report"])
    write_json(output_dir / "top_confusions.json", {"top_confusions": evaluation["top_confusions"]})
