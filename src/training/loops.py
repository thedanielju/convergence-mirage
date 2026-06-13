from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.nn.utils import clip_grad_norm_


# we bin predictions by confidence and measure how far each bin's average
# confidence sits from its empirical accuracy. we get this cheaply since the
# model already produced class probabilities.
def _expected_calibration_error(targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = predictions == targets
    ece = 0.0

    for lower in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        upper = lower + (1.0 / n_bins)
        if upper >= 1.0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        if not np.any(in_bin):
            continue
        bin_accuracy = float(np.mean(correct[in_bin]))
        bin_confidence = float(np.mean(confidences[in_bin]))
        ece += float(np.mean(in_bin)) * abs(bin_accuracy - bin_confidence)

    return float(ece)


# we track top-k accuracy for genre classification because near misses still
# reveal that the model ranked a musically plausible genre highly.
def _top_k_accuracy(targets: np.ndarray, probabilities: np.ndarray, k: int) -> float:
    if probabilities.shape[1] < k:
        return float("nan")
    top_k = np.argpartition(probabilities, kth=-k, axis=1)[:, -k:]
    return float(np.mean(np.any(top_k == targets[:, None], axis=1)))


# compute the common scalar classification metrics used by every model family.
# probabilities are optional so classical models that only expose hard labels
# can still share the same evaluation path.
def compute_classification_metrics(
    targets: list[int],
    predictions: list[int],
    probabilities: list[list[float]] | None = None,
) -> dict[str, float | None]:
    target_array = np.asarray(targets, dtype=np.int64)
    prediction_array = np.asarray(predictions, dtype=np.int64)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        average="weighted",
        zero_division=0,
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        target_array,
        prediction_array,
        average="micro",
        zero_division=0,
    )

    # sklearn warns when a tiny debug subset does not contain every predicted
    # class. that situation is expected for gates; the score is still defined.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced_accuracy = float(balanced_accuracy_score(target_array, prediction_array))

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(target_array, prediction_array)),
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "cohen_kappa": float(cohen_kappa_score(target_array, prediction_array)),
        "matthews_corrcoef": float(matthews_corrcoef(target_array, prediction_array)),
    }

    if probabilities is None:
        return metrics

    # we derive these from probabilities to gain visibility into ranking,
    # confidence, calibration, and separability without another forward pass.
    probability_array = np.asarray(probabilities, dtype=np.float64)
    confidences = np.max(probability_array, axis=1)
    predicted_confidences = probability_array[np.arange(probability_array.shape[0]), prediction_array]
    true_class_probabilities = probability_array[np.arange(probability_array.shape[0]), target_array]
    one_hot = np.eye(probability_array.shape[1], dtype=np.float64)[target_array]

    metrics.update(
        {
            "top_2_accuracy": _top_k_accuracy(target_array, probability_array, 2),
            "top_3_accuracy": _top_k_accuracy(target_array, probability_array, 3),
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "mean_predicted_class_confidence": float(np.mean(predicted_confidences)),
            "mean_true_class_probability": float(np.mean(true_class_probabilities)),
            "prediction_entropy": float(np.mean(-np.sum(probability_array * np.log(np.clip(probability_array, 1e-12, 1.0)), axis=1))),
            "expected_calibration_error": _expected_calibration_error(target_array, probability_array),
            "multiclass_brier": float(np.mean(np.sum((probability_array - one_hot) ** 2, axis=1))),
        }
    )

    # multiclass roc auc requires every class to be present in the evaluated
    # split. small debug subsets may omit rare classes, so auc is nullable.
    labels_present = np.unique(target_array)
    if labels_present.size == probability_array.shape[1]:
        try:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(target_array, probability_array, multi_class="ovr", average="macro"))
            metrics["roc_auc_ovr_weighted"] = float(
                roc_auc_score(target_array, probability_array, multi_class="ovr", average="weighted")
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = None
            metrics["roc_auc_ovr_weighted"] = None
    else:
        metrics["roc_auc_ovr_macro"] = None
        metrics["roc_auc_ovr_weighted"] = None

    return metrics


# extract the largest off-diagonal confusion pairs. we sort by true-class
# normalized rate before raw count so rare-class failures stay visible instead
# of being drowned out by rock/electronic support.
def top_confusions(
    confusion: np.ndarray,
    labels: list[int],
    label_names: dict[int, str] | None = None,
    top_n: int = 10,
) -> list[dict[str, object]]:
    rows = confusion.astype(np.float64)
    row_totals = rows.sum(axis=1, keepdims=True)
    normalized = np.divide(rows, row_totals, out=np.zeros_like(rows), where=row_totals != 0)

    pairs: list[dict[str, object]] = []
    for true_index, true_label in enumerate(labels):
        for predicted_index, predicted_label in enumerate(labels):
            if true_index == predicted_index:
                continue
            count = int(confusion[true_index, predicted_index])
            if count == 0:
                continue
            pairs.append(
                {
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "true_name": label_names.get(int(true_label), str(true_label)) if label_names else str(true_label),
                    "predicted_name": label_names.get(int(predicted_label), str(predicted_label)) if label_names else str(predicted_label),
                    "count": count,
                    "true_class_rate": float(normalized[true_index, predicted_index]),
                }
            )

    return sorted(pairs, key=lambda item: (item["true_class_rate"], item["count"]), reverse=True)[:top_n]


# run one full pass over a dataloader. passing an optimizer enables training;
# passing none switches to evaluation and disables gradient tracking.
def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, object]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []
    grad_norms: list[float] = []

    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.long, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # eval does not need autograd, which keeps validation/test memory low.
        grad_context = nullcontext() if is_training else torch.no_grad()
        with grad_context:
            # mixed precision reduces activation memory on cuda without changing
            # the model definition or checkpoint format.
            autocast_enabled = device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                logits = model(inputs)
                loss = criterion(logits, targets)

        if is_training:
            assert optimizer is not None
            assert scaler is not None
            # gradscaler avoids float16 gradient underflow during amp training.
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
                grad_norms.append(float(grad_norm.detach().cpu().item()))
            scaler.step(optimizer)
            scaler.update()

        # accumulate batch outputs so epoch-level metrics use every example.
        total_loss += float(loss.item()) * inputs.size(0)
        predictions = torch.argmax(logits, dim=1)
        probabilities = torch.softmax(logits.detach(), dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    metrics = compute_classification_metrics(all_targets, all_predictions, probabilities=all_probabilities)
    metrics.update(
        {
            "loss": float(total_loss / max(1, len(loader.dataset))),
            "targets": all_targets,
            "predictions": all_predictions,
            "probabilities": all_probabilities,
        }
    )
    if grad_norms:
        metrics["grad_norm_mean"] = float(np.mean(grad_norms))
        metrics["grad_norm_max"] = float(np.max(grad_norms))
        metrics["grad_norm_last"] = float(grad_norms[-1])
    return metrics


# compute test-time metrics, per-class reports, and confusion matrices needed
# for paper tables and error analysis.
def evaluate_predictions(
    targets: list[int],
    predictions: list[int],
    labels: list[int],
    probabilities: list[list[float]] | None = None,
    label_names: dict[int, str] | None = None,
) -> dict[str, object]:
    confusion = confusion_matrix(targets, predictions, labels=labels)
    normalized_confusion = confusion_matrix(targets, predictions, labels=labels, normalize="true")
    metrics = compute_classification_metrics(targets, predictions, probabilities=probabilities)
    metrics.update(
        {
            "confusion_matrix": confusion,
            "confusion_matrix_normalized": normalized_confusion,
            "top_confusions": top_confusions(confusion, labels=labels, label_names=label_names),
            "classification_report_text": classification_report(targets, predictions, labels=labels, zero_division=0),
            "classification_report": classification_report(
                targets,
                predictions,
                labels=labels,
                output_dict=True,
                zero_division=0,
            ),
        }
    )
    return metrics


# save model and optimizer state so a long run can be resumed from any epoch.
def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )
