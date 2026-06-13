from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import FMASpectrogramDataset
from src.models.cnn import CNN1D
from src.training.logging import (
    append_history_csv,
    append_jsonl,
    append_text_line,
    collect_cuda_stats,
    collect_environment,
    flatten_epoch_metrics,
    format_epoch_log,
    get_current_lr,
    get_grad_scaler_scale,
    save_evaluation_artifacts,
    write_json,
)
from src.training.loops import evaluate_predictions, run_epoch, save_checkpoint


# we load inverse-frequency class weights and normalize them to mean 1.0, which
# leaves the loss unchanged but keeps the weights easy to inspect.
def load_class_weights() -> torch.Tensor:
    payload = json.loads((REPO_ROOT / "data" / "splits" / "fma_medium" / "class_weights.json").read_text(encoding="utf-8"))
    weights = [float(payload["weights_by_id"][str(index)]) for index in range(16)]
    tensor = torch.tensor(weights, dtype=torch.float32)
    tensor = tensor / tensor.mean()
    return tensor


# we load the integer-to-genre mapping so confusion summaries can show names
# instead of only numeric class ids.
def load_label_names() -> dict[int, str]:
    payload = json.loads((REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json").read_text(encoding="utf-8"))
    return {int(index): str(label) for index, label in payload["id_to_label"].items()}


# debug gate 1: we try to overfit a single batch of 8 samples. if the model
# cannot reach perfect accuracy in 200 steps, the setup is broken and we stop.
def run_overfit_gate(model: CNN1D, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    dataset = FMASpectrogramDataset(split="training", apply_specaugment=False)
    subset = Subset(dataset, list(range(8)))
    loader = DataLoader(subset, batch_size=8, shuffle=True, num_workers=0)
    inputs, targets = next(iter(loader))
    inputs = inputs.to(device=device, dtype=torch.float32)
    targets = targets.to(device=device, dtype=torch.long)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    final_accuracy = 0.0
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        predictions = torch.argmax(logits, dim=1)
        final_accuracy = float((predictions == targets).float().mean().item())
        if final_accuracy >= 0.999:
            return {"passed": True, "steps": step, "final_accuracy": final_accuracy}

    return {"passed": False, "steps": 200, "final_accuracy": final_accuracy}


# debug gate 2: we train on 256 samples for 5 epochs and check that loss drops,
# which catches optimizer or pipeline bugs that only surface beyond one batch.
def run_subset_gate(model: CNN1D, criterion: nn.Module, device: torch.device) -> dict[str, object]:
    dataset = FMASpectrogramDataset(split="training", apply_specaugment=False)
    subset = Subset(dataset, list(range(256)))
    loader = DataLoader(subset, batch_size=32, shuffle=True, num_workers=0)
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    losses: list[float] = []
    for _ in range(5):
        epoch_metrics = run_epoch(model=model, loader=loader, criterion=criterion, optimizer=optimizer, device=device, scaler=scaler)
        losses.append(epoch_metrics["loss"])

    return {"passed": losses[-1] < losses[0], "losses": losses}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "config.json",
        {
            "model": "cnn1d",
            "batch_size": 32,
            "max_epochs": 40,
            "optimizer": "AdamW",
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "scheduler_t_max": 40,
            "early_stopping_metric": "val_macro_f1",
            "early_stopping_patience": 6,
            "amp": True,
            "specaugment": "training_split_only",
        },
    )

    # reduce allocator fragmentation before cuda memory is first requested.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # reserve roughly 25% of vram for the desktop and other processes. this
        # does not change the model, batch size, precision, optimizer, or data.
        torch.cuda.set_per_process_memory_fraction(0.75)
    write_json(args.output_dir / "environment.json", collect_environment())

    # class weights compensate for the extreme genre imbalance in fma medium.
    weights = load_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    label_names = load_label_names()

    overfit_model = CNN1D().to(device)
    overfit_gate = run_overfit_gate(model=overfit_model, criterion=criterion, device=device)
    if not overfit_gate["passed"]:
        raise RuntimeError(f"single-batch overfit gate failed: {json.dumps(overfit_gate)}")

    subset_model = CNN1D().to(device)
    subset_gate = run_subset_gate(model=subset_model, criterion=criterion, device=device)
    if not subset_gate["passed"]:
        raise RuntimeError(f"subset-loss gate failed: {json.dumps(subset_gate)}")
    write_json(
        args.output_dir / "debug_gates.json",
        {
            "single_batch_overfit": overfit_gate,
            "subset_loss_drop": subset_gate,
        },
    )

    # specaugment is applied only to the training split via the dataset default.
    train_dataset = FMASpectrogramDataset(split="training")
    val_dataset = FMASpectrogramDataset(split="validation", apply_specaugment=False)
    test_dataset = FMASpectrogramDataset(split="test", apply_specaugment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    model = CNN1D().to(device)
    # we use adamw so weight decay is decoupled from the gradient update, which
    # regularizes more cleanly than l2-penalized adam.
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # cosine annealing smoothly decays the learning rate toward zero by epoch 40.
    scheduler = CosineAnnealingLR(optimizer, T_max=40)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # early stopping tracks validation macro f1 because accuracy is dominated by
    # the majority genres. the best validation checkpoint is what gets tested.
    best_state = None
    best_epoch = 0
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    run_start = time.perf_counter()
    run_peak_memory_mb = 0.0

    for epoch in range(1, 41):
        # capture the lr before scheduler.step so the log records the rate used
        # for this epoch's optimizer updates.
        epoch_start = time.perf_counter()
        epoch_lr = get_current_lr(optimizer)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_metrics = run_epoch(model=model, loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, scaler=scaler)
        val_metrics = run_epoch(model=model, loader=val_loader, criterion=criterion, optimizer=None, device=device, scaler=None)
        scheduler.step()

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_seconds = time.perf_counter() - epoch_start
        cuda_stats = collect_cuda_stats(device)
        if cuda_stats["cuda_peak_memory_allocated_mb"] is not None:
            run_peak_memory_mb = max(run_peak_memory_mb, float(cuda_stats["cuda_peak_memory_allocated_mb"]))
        # history.csv is flat for plotting, history.jsonl preserves the same
        # row in a structured format, and train.log stays readable live.
        row = {
            "epoch": epoch,
            "max_epochs": 40,
            "learning_rate": epoch_lr,
            "grad_scaler_scale": get_grad_scaler_scale(scaler),
            "epoch_seconds": epoch_seconds,
            "samples_per_sec": float((len(train_dataset) + len(val_dataset)) / max(epoch_seconds, 1e-9)),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "patience": 6,
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

        if epochs_without_improvement >= 6:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"epoch": best_epoch, "model_state_dict": best_state}, args.output_dir / "best.pt")

    test_metrics = run_epoch(model=model, loader=test_loader, criterion=criterion, optimizer=None, device=device, scaler=None)
    evaluation = evaluate_predictions(
        test_metrics["targets"],
        test_metrics["predictions"],
        labels=list(range(16)),
        probabilities=test_metrics["probabilities"],
        label_names=label_names,
    )
    save_evaluation_artifacts(args.output_dir, evaluation)

    metrics_payload = {
        "debug_gates": {
            "single_batch_overfit": overfit_gate,
            "subset_loss_drop": subset_gate,
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
        "test_top_2_accuracy": evaluation["top_2_accuracy"],
        "test_top_3_accuracy": evaluation["top_3_accuracy"],
        "test_expected_calibration_error": evaluation["expected_calibration_error"],
        "test_multiclass_brier": evaluation["multiclass_brier"],
        "test_roc_auc_ovr_macro": evaluation["roc_auc_ovr_macro"],
        "classification_report": evaluation["classification_report"],
        "top_confusions": evaluation["top_confusions"],
    }
    write_json(args.output_dir / "metrics.json", metrics_payload)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
