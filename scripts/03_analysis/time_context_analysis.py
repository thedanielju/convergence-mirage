from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.representations import infer_model_name, load_checkpoint_model
from src.data.dataset import FMASpectrogramDataset


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def limited_dataset(split: str, limit: int | None, data_root: Path | None, crop_seconds: float) -> torch.utils.data.Dataset:
    dataset = FMASpectrogramDataset(split=split, apply_specaugment=False, crop_seconds=crop_seconds, processed_dir=data_root)
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(int(limit), len(dataset)))))


def mean_transformer_cls_attention(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    encoded = model._prepare_tokens(inputs)
    layer_curves: list[torch.Tensor] = []
    for layer in model.encoder.layers:
        if getattr(layer, "norm_first", False):
            attn_input = layer.norm1(encoded)
            attn_output, attn_weights = layer.self_attn(
                attn_input,
                attn_input,
                attn_input,
                need_weights=True,
                average_attn_weights=False,
            )
            encoded = encoded + layer.dropout1(attn_output)
            encoded = encoded + layer._ff_block(layer.norm2(encoded))
        else:
            attn_output, attn_weights = layer.self_attn(
                encoded,
                encoded,
                encoded,
                need_weights=True,
                average_attn_weights=False,
            )
            encoded = layer.norm1(encoded + layer.dropout1(attn_output))
            encoded = layer.norm2(encoded + layer._ff_block(encoded))
        # attn_weights is batch x heads x target_tokens x source_tokens, and we
        # drop the cls source column so the curve covers only the time frames.
        layer_curves.append(attn_weights[:, :, 0, 1:].detach().float().mean(dim=(0, 1)).cpu())
    return torch.stack(layer_curves, dim=0).mean(dim=0).numpy()


def mean_mamba_activation_magnitude(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    tokens = inputs.transpose(1, 2)
    tokens = model.input_projection(tokens)
    tokens = model.input_dropout(tokens)
    layer_curves: list[torch.Tensor] = []
    for block in model.blocks:
        tokens = block(tokens)
        layer_curves.append(tokens.detach().float().norm(dim=-1).mean(dim=0).cpu())
    return torch.stack(layer_curves, dim=0).mean(dim=0).numpy()


def analyze_run(
    run_dir: Path,
    output_dir: Path,
    data_root: Path | None,
    crop_seconds: float,
    split: str,
    limit_samples: int,
    batch_size: int,
    num_workers: int,
    device: str | None,
) -> dict[str, Any]:
    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    model_name = infer_model_name(checkpoint, run_dir=run_dir)
    if model_name not in {"transformer", "mamba1", "mamba2"}:
        raise ValueError(f"time-context analysis only supports transformer/mamba models, got {model_name}: {run_dir}")
    torch_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint_model(checkpoint, model_name=model_name, device=torch_device, strict=True)
    dataset = limited_dataset(split=split, limit=limit_samples, data_root=data_root, crop_seconds=crop_seconds)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch_device.type == "cuda")

    curves: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.no_grad():
        for inputs, _targets in loader:
            batch = inputs.to(device=torch_device, dtype=torch.float32, non_blocking=True)
            if model_name == "transformer":
                curves.append(mean_transformer_cls_attention(model, batch))
            else:
                curves.append(mean_mamba_activation_magnitude(model, batch))
    curve = np.mean(np.stack(curves, axis=0), axis=0).astype(np.float64)
    if not np.isfinite(curve).all():
        raise ValueError(f"non-finite time-context curve for {run_dir}")
    if curve.sum() > 0:
        normalized = curve / curve.sum()
    else:
        normalized = curve

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = run_dir.name
    csv_path = output_dir / f"{safe_name}_time_context.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_index", "seconds", "value", "normalized_value"])
        writer.writeheader()
        for index, (value, norm_value) in enumerate(zip(curve.tolist(), normalized.tolist())):
            writer.writerow(
                {
                    "frame_index": index,
                    "seconds": index * 512 / 22050,
                    "value": value,
                    "normalized_value": norm_value,
                }
            )

    figure_path = output_dir / f"{safe_name}_time_context.png"
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(curve.shape[0]) * 512 / 22050, normalized)
    ylabel = "CLS attention share" if model_name == "transformer" else "activation magnitude share"
    plt.xlabel("Time (seconds)")
    plt.ylabel(ylabel)
    plt.title(f"{safe_name}: {model_name} time-context profile")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    return {
        "run_dir": str(run_dir),
        "model": model_name,
        "checkpoint": str(checkpoint),
        "split": split,
        "sample_count": len(dataset),
        "crop_seconds": crop_seconds,
        "data_root": str(data_root) if data_root is not None else "data/processed/fma_medium",
        "curve_length": int(curve.shape[0]),
        "csv": str(csv_path),
        "figure": str(figure_path),
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": "Transformer values are mean CLS-to-time attention. Mamba values are residual-block token activation norm, an activation/state-magnitude proxy rather than literal hidden SSM state.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-of-context analysis for Transformer and Mamba music classifiers.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "time_context_analysis")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--crop-seconds", type=float, default=60.0)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--limit-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    args = parser.parse_args()

    results = [
        analyze_run(
            run_dir=run_dir,
            output_dir=args.output_dir,
            data_root=args.data_root,
            crop_seconds=args.crop_seconds,
            split=args.split,
            limit_samples=args.limit_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        for run_dir in args.run_dir
    ]
    manifest = {
        "created_at_utc": utc_now(),
        "entrypoint": "scripts/03_analysis/time_context_analysis.py",
        "command": " ".join(sys.argv),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
