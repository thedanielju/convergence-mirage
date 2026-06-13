from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from src.data.dataset import FMASpectrogramDataset
from src.models.cnn import CNN1D, CNN2D
from src.models.lstm import BiLSTMClassifier
from src.models.mamba import Mamba1Classifier, Mamba2Classifier, check_mamba_dependencies
from src.models.transformer import TransformerClassifier
from src.training.logging import (
    append_history_csv,
    append_jsonl,
    append_text_line,
    collect_cuda_stats,
    ARTIFACT_SCHEMA_VERSION,
    collect_environment,
    flatten_epoch_metrics,
    format_epoch_log,
    get_current_lr,
    get_grad_scaler_scale,
    prepare_output_dir,
    save_evaluation_artifacts,
    write_json,
)
from src.training.loops import evaluate_predictions, run_epoch, save_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_MAP_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json"
CLASS_WEIGHTS_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "class_weights.json"


@dataclass(frozen=True)
class DeepModelConfig:
    model: str
    display_name: str
    learning_rate: float
    weight_decay: float
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    warmup_epochs: int
    max_grad_norm: float | None
    overfit_steps: int
    subset_size: int
    subset_epochs: int
    architecture: dict[str, Any]


MODEL_DEFAULTS: dict[str, DeepModelConfig] = {
    "cnn": DeepModelConfig(
        model="cnn",
        display_name="cnn1d",
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.0,
        batch_size=32,
        max_epochs=40,
        patience=6,
        warmup_epochs=0,
        max_grad_norm=None,
        overfit_steps=200,
        subset_size=256,
        subset_epochs=5,
        architecture={
            "input_shape": [128, 1292],
            "channels": [128, 128, 256, 512, 512],
            "kernels": [7, 5, 5, 5],
            "pooling": "global_average",
            "representation_dim": 512,
        },
    ),
    "cnn2d": DeepModelConfig(
        model="cnn2d",
        display_name="cnn2d_final_baseline",
        learning_rate=3e-4,
        weight_decay=1e-4,
        dropout=0.0,
        batch_size=32,
        max_epochs=40,
        patience=6,
        warmup_epochs=0,
        max_grad_norm=None,
        overfit_steps=200,
        subset_size=256,
        subset_epochs=5,
        architecture={
            "input_shape": [128, 1292],
            "internal_input_shape": [1, 128, 1292],
            "channels": [1, 32, 128, 256, 512, 512],
            "kernels": [[5, 7], [3, 5], [3, 5], [3, 3], [3, 3]],
            "time_frequency_pooling": [[2, 4], [2, 4], [2, 2], [2, 2], None],
            "pooling": "adaptive_average_2d",
            "representation_dim": 512,
            "classifier": "Linear(512, 16)",
            "parameter_target": "about 4.1M parameters; within 2-5M target",
        },
    ),
    "lstm": DeepModelConfig(
        model="lstm",
        display_name="bilstm_attention",
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.3,
        batch_size=32,
        max_epochs=50,
        patience=10,
        warmup_epochs=5,
        max_grad_norm=1.0,
        overfit_steps=250,
        subset_size=256,
        subset_epochs=5,
        architecture={
            "input_shape": [128, 1292],
            "hidden_dim": 256,
            "num_layers": 2,
            "bidirectional": True,
            "pooling": "attention_weighted",
            "representation_dim": 512,
        },
    ),
    "transformer": DeepModelConfig(
        model="transformer",
        display_name="transformer_classifier",
        learning_rate=5e-4,
        weight_decay=1e-4,
        dropout=0.1,
        batch_size=32,
        max_epochs=50,
        patience=10,
        warmup_epochs=5,
        max_grad_norm=1.0,
        overfit_steps=300,
        subset_size=256,
        subset_epochs=8,
        architecture={
            "input_shape": [128, 1292],
            "d_model": 256,
            "nhead": 4,
            "num_layers": 4,
            "dim_feedforward": 1024,
            "pooling": "cls_token",
            "representation_dim": 256,
        },
    ),
    "mamba1": DeepModelConfig(
        model="mamba1",
        display_name="mamba1_classifier",
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        batch_size=32,
        max_epochs=50,
        patience=10,
        warmup_epochs=5,
        max_grad_norm=1.0,
        overfit_steps=250,
        subset_size=256,
        subset_epochs=5,
        architecture={
            "input_shape": [128, 1292],
            "d_model": 256,
            "num_layers": 5,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "pooling": "mean_temporal",
            "representation_dim": 256,
        },
    ),
    "mamba2": DeepModelConfig(
        model="mamba2",
        display_name="mamba2_classifier",
        learning_rate=1e-3,
        weight_decay=1e-4,
        dropout=0.1,
        batch_size=32,
        max_epochs=50,
        patience=10,
        warmup_epochs=5,
        max_grad_norm=1.0,
        overfit_steps=250,
        subset_size=256,
        subset_epochs=5,
        architecture={
            "input_shape": [128, 1292],
            "d_model": 256,
            "num_layers": 5,
            "d_state": 64,
            "d_conv": 4,
            "expand": 2,
            "pooling": "mean_temporal",
            "representation_dim": 256,
        },
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_class_weights() -> torch.Tensor:
    payload = json.loads(CLASS_WEIGHTS_PATH.read_text(encoding="utf-8"))
    weights = [float(payload["weights_by_id"][str(index)]) for index in range(16)]
    tensor = torch.tensor(weights, dtype=torch.float32)
    return tensor / tensor.mean()


def load_label_names() -> dict[int, str]:
    payload = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    return {int(index): str(label) for index, label in payload["id_to_label"].items()}


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_config(args: Any) -> DeepModelConfig:
    config = MODEL_DEFAULTS[args.model]
    return replace(
        config,
        learning_rate=args.learning_rate if args.learning_rate is not None else config.learning_rate,
        dropout=args.dropout if args.dropout is not None else config.dropout,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        max_epochs=args.max_epochs if args.max_epochs is not None else config.max_epochs,
        patience=args.patience if args.patience is not None else config.patience,
        overfit_steps=args.overfit_steps if args.overfit_steps is not None else config.overfit_steps,
        subset_size=args.gate_subset_size if args.gate_subset_size is not None else config.subset_size,
        subset_epochs=args.gate_subset_epochs if args.gate_subset_epochs is not None else config.subset_epochs,
    )


def build_model(model_name: str, dropout: float) -> nn.Module:
    if model_name == "cnn":
        return CNN1D()
    if model_name == "cnn2d":
        return CNN2D()
    if model_name == "lstm":
        return BiLSTMClassifier(dropout=dropout)
    if model_name == "transformer":
        return TransformerClassifier(dropout=dropout)
    if model_name == "mamba1":
        return Mamba1Classifier(dropout=dropout)
    if model_name == "mamba2":
        return Mamba2Classifier(dropout=dropout)
    raise ValueError(f"unsupported model: {model_name}")


# we chain a linear warmup into cosine decay with sequentiallr so the warmup
# owns its first epochs and cosine anneals over the remaining budget. the
# attention-based models need the warmup to avoid an early-epoch loss spike.
def build_scheduler(optimizer: torch.optim.Optimizer, config: DeepModelConfig) -> torch.optim.lr_scheduler.LRScheduler:
    if config.warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(1, config.max_epochs))

    warmup_epochs = min(config.warmup_epochs, max(1, config.max_epochs - 1))
    linear_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_decay = CosineAnnealingLR(optimizer, T_max=max(1, config.max_epochs - warmup_epochs))
    return SequentialLR(optimizer, schedulers=[linear_warmup, cosine_decay], milestones=[warmup_epochs])


def limited_dataset(dataset: torch.utils.data.Dataset, limit: int | None) -> torch.utils.data.Dataset:
    if limit is None:
        return dataset
    count = min(int(limit), len(dataset))
    return Subset(dataset, list(range(count)))


def build_dataloaders(args: Any, config: DeepModelConfig, device: torch.device) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset, DataLoader, DataLoader, DataLoader]:
    train_apply_specaugment = False if getattr(args, "disable_specaugment", False) else None
    train_dataset = limited_dataset(FMASpectrogramDataset(split="training", apply_specaugment=train_apply_specaugment, crop_seconds=args.crop_seconds, processed_dir=args.data_root), args.limit_train_samples)
    val_dataset = limited_dataset(FMASpectrogramDataset(split="validation", apply_specaugment=False, crop_seconds=args.crop_seconds, processed_dir=args.data_root), args.limit_val_samples)
    test_dataset = limited_dataset(FMASpectrogramDataset(split="test", apply_specaugment=False, crop_seconds=args.crop_seconds, processed_dir=args.data_root), args.limit_test_samples)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if args.num_workers > 0 else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=make_generator(args.seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, generator=make_generator(args.seed + 1), **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, generator=make_generator(args.seed + 2), **loader_kwargs)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def dataset_track_ids(dataset: torch.utils.data.Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        base_ids = dataset_track_ids(dataset.dataset)
        return [base_ids[int(index)] for index in dataset.indices]
    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError("dataset does not expose track records for test_track_ids.npy")
    return [int(record["track_id"]) for record in records]


def write_run_manifest(output_dir: Path, payload: dict[str, Any]) -> None:
    artifact_names = [
        "config.json",
        "environment.json",
        "debug_gates.json",
        "history.csv",
        "history.jsonl",
        "train.log",
        "best.pt",
        "metrics.json",
        "classification_report.json",
        "report.txt",
        "confusion_matrix.npy",
        "confusion_matrix_normalized.npy",
        "top_confusions.json",
        "predictions.npy",
        "probabilities.npy",
        "test_labels.npy",
        "test_track_ids.npy",
        "run_manifest.json",
    ]
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "updated_at_utc": utc_now(),
        "artifact_paths": {name: str(output_dir / name) for name in artifact_names},
        **payload,
    }
    write_json(output_dir / "run_manifest.json", manifest)


def make_config_payload(args: Any, config: DeepModelConfig) -> dict[str, Any]:
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model": config.model,
        "display_name": config.display_name,
        "seed": args.seed,
        "entrypoint": "scripts/02_train/train_deep.py",
        "api_contract": {
            "forward": "forward(inputs) -> logits",
            "penultimate": "get_penultimate(inputs) -> representation",
            "intermediate": "get_intermediate_representations(inputs) -> dict[str, Tensor]",
        },
        "training": {
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "optimizer": "AdamW",
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "scheduler": "CosineAnnealingLR" if config.warmup_epochs <= 0 else "SequentialLR(LinearLR warmup + CosineAnnealingLR)",
            "warmup_epochs": config.warmup_epochs,
            "early_stopping_metric": "val_macro_f1",
            "early_stopping_patience": config.patience,
            "max_grad_norm": config.max_grad_norm,
            "amp": {"enabled_on_cuda": True, "enabled_on_cpu": False},
            "specaugment": "disabled" if getattr(args, "disable_specaugment", False) else "training_split_only",
            "class_weights": "data/splits/fma_medium/class_weights.json normalized to mean 1.0",
        },
        "debug_gates": {
            "single_batch_overfit_steps": config.overfit_steps,
            "single_batch_size": args.gate_batch_size,
            "subset_size": config.subset_size,
            "subset_epochs": config.subset_epochs,
        },
        "architecture": config.architecture,
        "data_loader": {
            "num_workers": args.num_workers,
            "seeded_generator": True,
            "worker_init_fn": "seed_worker" if args.num_workers > 0 else None,
            "crop_seconds": args.crop_seconds,
            "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium",
        },
        "crop_seconds": args.crop_seconds,
        "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium",
        "data_limits": {
            "limit_train_samples": args.limit_train_samples,
            "limit_val_samples": args.limit_val_samples,
            "limit_test_samples": args.limit_test_samples,
        },
    }


def run_forward_smoke(
    model_name: str,
    config: DeepModelConfig,
    criterion: nn.Module,
    device: torch.device,
    seed: int,
    crop_seconds: float,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    dataset = FMASpectrogramDataset(split="training", apply_specaugment=False, crop_seconds=crop_seconds, processed_dir=data_root)
    loader = DataLoader(Subset(dataset, [0, 1]), batch_size=2, shuffle=False, num_workers=0, generator=make_generator(seed))
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)
    model = build_model(model_name, config.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits = model(inputs)
        loss = criterion(logits, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return {
        "passed": list(logits.shape) == [2, 16],
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu().item()),
        "parameter_count": count_parameters(model),
    }


def run_mamba_dependency_gate(model_name: str, config: DeepModelConfig, device: torch.device) -> dict[str, Any]:
    gate = check_mamba_dependencies(model_name)
    if not gate.get("passed", False):
        gate["forward_smoke"] = {"passed": False, "skipped": True, "reason": gate.get("reason")}
        return gate

    try:
        model = build_model(model_name, config.dropout).to(device)
        dummy = torch.randn(2, 128, 1292, device=device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(dummy)
        gate["forward_smoke"] = {
            "passed": list(logits.shape) == [2, 16],
            "logits_shape": list(logits.shape),
            "parameter_count": count_parameters(model),
        }
        gate["passed"] = bool(gate["forward_smoke"]["passed"])
    except Exception as exc:  # pragma: no cover - depends on optional cuda extension build
        gate["passed"] = False
        gate["reason"] = f"mamba build/import forward smoke failed: {exc}"
        gate["forward_smoke"] = {"passed": False, "error": str(exc)}
    return gate


def run_overfit_gate(
    model_name: str,
    config: DeepModelConfig,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int,
    seed: int,
    crop_seconds: float,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    dataset = FMASpectrogramDataset(split="training", apply_specaugment=False, crop_seconds=crop_seconds, processed_dir=data_root)
    subset = Subset(dataset, list(range(batch_size)))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0, generator=make_generator(seed + 10))
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)

    model = build_model(model_name, config.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=max(config.learning_rate, 1e-3), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    final_accuracy = 0.0
    final_loss = None
    for step in range(1, config.overfit_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if config.max_grad_norm is not None:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        predictions = torch.argmax(logits, dim=1)
        final_accuracy = float((predictions == targets).float().mean().item())
        final_loss = float(loss.detach().cpu().item())
        if final_accuracy >= 0.999:
            return {"passed": True, "steps": step, "final_accuracy": final_accuracy, "final_loss": final_loss}

    return {"passed": False, "steps": config.overfit_steps, "final_accuracy": final_accuracy, "final_loss": final_loss}


def run_subset_gate(
    model_name: str,
    config: DeepModelConfig,
    criterion: nn.Module,
    device: torch.device,
    seed: int,
    crop_seconds: float,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    dataset = FMASpectrogramDataset(split="training", apply_specaugment=False, crop_seconds=crop_seconds, processed_dir=data_root)
    subset_size = min(config.subset_size, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=config.batch_size, shuffle=True, num_workers=0, generator=make_generator(seed + 20))
    model = build_model(model_name, config.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    losses: list[float] = []
    for _ in range(config.subset_epochs):
        epoch_metrics = run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=config.max_grad_norm,
        )
        losses.append(float(epoch_metrics["loss"]))

    return {"passed": len(losses) >= 2 and losses[-1] < losses[0], "losses": losses, "subset_size": subset_size}


def gates_passed(gates: dict[str, Any]) -> bool:
    for gate in gates.values():
        if isinstance(gate, dict) and gate.get("passed") is False:
            return False
    return True


# we gate every full run behind cheap sanity checks: the model must overfit a
# single batch and drop loss on a small subset. if these fail the architecture
# or data pipeline is broken, so we stop before spending gpu hours.
def run_debug_gates(args: Any, config: DeepModelConfig, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    if config.model.startswith("mamba"):
        gates["mamba_dependency_build"] = run_mamba_dependency_gate(config.model, config, device)
        if not gates["mamba_dependency_build"].get("passed", False):
            gates["single_batch_overfit"] = {"passed": False, "skipped": True, "reason": "mamba dependency/build gate failed"}
            gates["subset_loss_drop"] = {"passed": False, "skipped": True, "reason": "mamba dependency/build gate failed"}
            return gates

    gates["single_batch_overfit"] = run_overfit_gate(
        model_name=config.model,
        config=config,
        criterion=criterion,
        device=device,
        batch_size=args.gate_batch_size,
        seed=args.seed,
        crop_seconds=args.crop_seconds,
        data_root=args.data_root,
    )
    if gates["single_batch_overfit"].get("passed", False):
        gates["subset_loss_drop"] = run_subset_gate(config.model, config, criterion, device, seed=args.seed, crop_seconds=args.crop_seconds, data_root=args.data_root)
    else:
        gates["subset_loss_drop"] = {"passed": False, "skipped": True, "reason": "single-batch overfit gate failed"}
    return gates


def train_full_run(args: Any, config: DeepModelConfig, criterion: nn.Module, label_names: dict[int, str], device: torch.device, debug_gates: dict[str, Any]) -> dict[str, Any]:
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_dataloaders(args, config, device)
    model = build_model(config.model, config.dropout).to(device)
    parameter_count = count_parameters(model)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    run_start = time.perf_counter()
    run_peak_memory_mb = 0.0

    for epoch in range(1, config.max_epochs + 1):
        epoch_start = time.perf_counter()
        epoch_lr = get_current_lr(optimizer)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=config.max_grad_norm,
        )
        val_metrics = run_epoch(model=model, loader=val_loader, criterion=criterion, optimizer=None, device=device, scaler=None)
        scheduler.step()

        # we select on validation macro-f1 rather than accuracy because the genre
        # imbalance makes accuracy reward the majority classes. the best state is
        # snapshotted to cpu so a later epoch cannot overwrite it before stopping.
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start
        cuda_stats = collect_cuda_stats(device)
        if cuda_stats["cuda_peak_memory_allocated_mb"] is not None:
            run_peak_memory_mb = max(run_peak_memory_mb, float(cuda_stats["cuda_peak_memory_allocated_mb"]))

        row = {
            "epoch": epoch,
            "max_epochs": config.max_epochs,
            "model": config.model,
            "seed": args.seed,
            "learning_rate": epoch_lr,
            "grad_scaler_scale": get_grad_scaler_scale(scaler),
            "epoch_seconds": epoch_seconds,
            "samples_per_sec": float((len(train_dataset) + len(val_dataset)) / max(epoch_seconds, 1e-9)),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "patience": config.patience,
            "patience_used": epochs_without_improvement,
            **flatten_epoch_metrics(train_metrics, "train"),
            **flatten_epoch_metrics(val_metrics, "val"),
            **cuda_stats,
        }
        log_line = format_epoch_log(row)
        append_history_csv(args.output_dir / "history.csv", row)
        append_jsonl(args.output_dir / "history.jsonl", row)
        append_text_line(args.output_dir / "train.log", log_line)
        print(log_line)

        save_checkpoint(
            checkpoint_path=args.output_dir / f"checkpoint_epoch_{epoch:02d}.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics=row,
        )

        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("no best checkpoint was selected; max_epochs must be at least 1")

    model.load_state_dict(best_state)
    torch.save(
        {
            "epoch": best_epoch,
            "model": config.model,
            "seed": args.seed,
            "model_state_dict": best_state,
            "parameter_count": parameter_count,
            "config": asdict(config),
        },
        args.output_dir / "best.pt",
    )

    test_metrics = run_epoch(model=model, loader=test_loader, criterion=criterion, optimizer=None, device=device, scaler=None)
    evaluation = evaluate_predictions(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(16)),
        probabilities=test_metrics["probabilities"],
        label_names=label_names,
    )
    save_evaluation_artifacts(args.output_dir, evaluation)
    np.save(args.output_dir / "predictions.npy", np.asarray(test_metrics["predictions"], dtype=np.int64), allow_pickle=False)
    np.save(args.output_dir / "probabilities.npy", np.asarray(test_metrics["probabilities"], dtype=np.float32), allow_pickle=False)
    np.save(args.output_dir / "test_labels.npy", np.asarray(test_metrics["targets"], dtype=np.int64), allow_pickle=False)
    np.save(args.output_dir / "test_track_ids.npy", np.asarray(dataset_track_ids(test_dataset), dtype=np.int64), allow_pickle=False)

    return {
        "model": config.model,
        "seed": args.seed,
        "debug_gates": debug_gates,
        "parameter_count": parameter_count,
        "dataset_sizes": {
            "training": len(train_dataset),
            "validation": len(val_dataset),
            "test": len(test_dataset),
        },
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "total_train_seconds": float(time.perf_counter() - run_start),
        "peak_gpu_memory_allocated_mb": run_peak_memory_mb if device.type == "cuda" else None,
        "test_accuracy": evaluation["accuracy"],
        "test_balanced_accuracy": evaluation["balanced_accuracy"],
        "test_macro_f1": evaluation["macro_f1"],
        "test_weighted_f1": evaluation["weighted_f1"],
        "test_macro_precision": evaluation["macro_precision"],
        "test_macro_recall": evaluation["macro_recall"],
        "test_cohen_kappa": evaluation["cohen_kappa"],
        "test_matthews_corrcoef": evaluation["matthews_corrcoef"],
        "test_top_2_accuracy": evaluation.get("top_2_accuracy"),
        "test_top_3_accuracy": evaluation.get("top_3_accuracy"),
        "test_expected_calibration_error": evaluation.get("expected_calibration_error"),
        "test_multiclass_brier": evaluation.get("multiclass_brier"),
        "test_roc_auc_ovr_macro": evaluation.get("roc_auc_ovr_macro"),
        "classification_report": evaluation["classification_report"],
        "top_confusions": evaluation["top_confusions"],
    }


def run_from_args(args: Any) -> int:
    if args.max_epochs is not None and args.max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1")
    prepare_output_dir(args.output_dir, overwrite=args.overwrite)
    config = resolve_config(args)
    set_seed(args.seed)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256,expandable_segments:True")
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    write_json(args.output_dir / "config.json", make_config_payload(args, config))
    write_json(args.output_dir / "environment.json", collect_environment())
    write_run_manifest(
        args.output_dir,
        {
            "status": "running",
            "model": config.model,
            "seed": args.seed,
            "crop_seconds": args.crop_seconds,
            "amp_enabled": device.type == "cuda",
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "cuda_memory": collect_cuda_stats(device),
            "backend_flags": collect_environment().get("backend_flags"),
            "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium",
            "command": " ".join(sys.argv),
            "started_at_utc": utc_now(),
        },
    )

    criterion = nn.CrossEntropyLoss(weight=load_class_weights().to(device))
    label_names = load_label_names()

    if args.forward_smoke_only:
        gates: dict[str, Any] = {}
        if config.model.startswith("mamba"):
            gates["mamba_dependency_build"] = run_mamba_dependency_gate(config.model, config, device)
            if not gates["mamba_dependency_build"].get("passed", False):
                payload = {"model": config.model, "seed": args.seed, "debug_gates": gates}
                write_json(args.output_dir / "debug_gates.json", gates)
                write_json(args.output_dir / "metrics.json", payload)
                write_run_manifest(args.output_dir, {"status": "blocked", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium", "blocked_reason": gates["mamba_dependency_build"].get("reason")})
                print(json.dumps(payload, indent=2))
                return 2
        else:
            gates["forward_smoke"] = run_forward_smoke(config.model, config, criterion, device, seed=args.seed, crop_seconds=args.crop_seconds, data_root=args.data_root)
        write_json(args.output_dir / "debug_gates.json", gates)
        write_json(args.output_dir / "metrics.json", {"model": config.model, "seed": args.seed, "debug_gates": gates})
        write_run_manifest(args.output_dir, {"status": "forward_smoke_completed", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium"})
        print(json.dumps({"model": config.model, "seed": args.seed, "debug_gates": gates}, indent=2))
        return 0 if gates_passed(gates) else 2

    if args.skip_debug_gates:
        debug_gates: dict[str, Any] = {"skipped": True, "reason": "--skip-debug-gates was set"}
        if config.model.startswith("mamba"):
            debug_gates["mamba_dependency_build"] = run_mamba_dependency_gate(config.model, config, device)
            if not debug_gates["mamba_dependency_build"].get("passed", False):
                payload = {"model": config.model, "seed": args.seed, "debug_gates": debug_gates}
                write_json(args.output_dir / "debug_gates.json", debug_gates)
                write_json(args.output_dir / "metrics.json", payload)
                write_run_manifest(args.output_dir, {"status": "blocked", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium", "blocked_reason": debug_gates["mamba_dependency_build"].get("reason")})
                print(json.dumps(payload, indent=2))
                return 2
    else:
        debug_gates = run_debug_gates(args, config, criterion, device)

    write_json(args.output_dir / "debug_gates.json", debug_gates)
    if not gates_passed(debug_gates):
        payload = {"model": config.model, "seed": args.seed, "debug_gates": debug_gates}
        write_json(args.output_dir / "metrics.json", payload)
        write_run_manifest(args.output_dir, {"status": "blocked", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium", "blocked_reason": "debug gate failed"})
        print(json.dumps(payload, indent=2))
        return 2

    if args.debug_gates_only:
        payload = {"model": config.model, "seed": args.seed, "debug_gates": debug_gates}
        write_json(args.output_dir / "metrics.json", payload)
        write_run_manifest(args.output_dir, {"status": "debug_gates_completed", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium"})
        print(json.dumps(payload, indent=2))
        return 0

    metrics_payload = train_full_run(args, config, criterion, label_names, device, debug_gates)
    write_json(args.output_dir / "metrics.json", metrics_payload)
    write_run_manifest(args.output_dir, {"status": "completed", "model": config.model, "seed": args.seed, "crop_seconds": args.crop_seconds, "data_root": str(args.data_root) if args.data_root is not None else "data/processed/fma_medium", "finished_at_utc": utc_now()})
    print(json.dumps(metrics_payload, indent=2))
    return 0
