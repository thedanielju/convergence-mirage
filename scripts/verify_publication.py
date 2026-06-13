from __future__ import annotations

import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required publication artifact is missing: {relative_path}")
    return path


def require_json(relative_path: str) -> object:
    path = require_file(relative_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_csv_rows(relative_path: str, *, minimum_rows: int = 1) -> None:
    path = require_file(relative_path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < minimum_rows:
        raise ValueError(f"{relative_path} has {len(rows)} data rows; expected {minimum_rows}+")


def verify_artifacts() -> None:
    """Check the compact result surface used by the README and paper narrative."""

    headline = require_json("results/aggregated/headline_metrics.json")
    if not isinstance(headline, dict) or not headline:
        raise ValueError("headline_metrics.json must be a non-empty JSON object")

    paired_tests = require_json("results/aggregated/paired_tests.json")
    if not isinstance(paired_tests, dict):
        raise ValueError("paired_tests.json must be a JSON object")

    cka_summary = require_json("results/cka/novelty_gap.json")
    if not isinstance(cka_summary, dict):
        raise ValueError("novelty_gap.json must be a JSON object")

    require_csv_rows("results/paper_assets/table1_performance.csv", minimum_rows=3)
    require_csv_rows("results/paper_assets/table2_efficiency_calibration.csv", minimum_rows=3)
    require_csv_rows("results/paper_assets/table3_top_confusions.csv", minimum_rows=3)
    require_file("docs/paper/convergence-mirage-paper.pdf")


def verify_model_smoke() -> None:
    """Run tiny CPU forward passes through the required model families.

    The goal is not to reproduce training quality.  This catches broken imports,
    shape drift, and package-layout mistakes in a way that runs without raw FMA
    audio, processed spectrogram caches, checkpoints, or a GPU.
    """

    try:
        import torch
    except ModuleNotFoundError:
        print("[optional] PyTorch is not installed; model forward smoke checks skipped")
        return

    from src.models.cnn import CNN2D
    from src.models.lstm import BiLSTMClassifier
    from src.models.transformer import TransformerClassifier
    from src.models.mamba import check_mamba_dependencies

    sample = torch.randn(2, 128, 64)

    models = {
        "cnn2d": CNN2D(num_classes=16),
        "bilstm": BiLSTMClassifier(
            num_classes=16,
            hidden_dim=16,
            num_layers=1,
            attention_dim=8,
            dropout=0.0,
        ),
        "transformer": TransformerClassifier(
            num_classes=16,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            max_len=128,
        ),
    }

    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            logits = model(sample)
            if tuple(logits.shape) != (2, 16):
                raise ValueError(f"{name} produced logits shape {tuple(logits.shape)}")

    for version in ("mamba1", "mamba2"):
        status = check_mamba_dependencies(version)
        if not status["passed"]:
            print(f"[optional] {version} dependency check skipped: {status.get('reason')}")


def main() -> None:
    verify_artifacts()
    verify_model_smoke()
    print("publication verification passed")


if __name__ == "__main__":
    main()
