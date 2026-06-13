from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.deep import MODEL_DEFAULTS, run_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="shared deep-model training entrypoint for FMA medium")
    parser.add_argument("--model", choices=sorted(MODEL_DEFAULTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing non-empty output directory")
    parser.add_argument("--debug-gates-only", action="store_true", help="run dependency/debug gates and exit before full training")
    parser.add_argument("--forward-smoke-only", action="store_true", help="run a tiny forward/backward smoke check and exit")
    parser.add_argument("--skip-debug-gates", action="store_true", help="skip debug gates before full training; mamba dependency gate still runs")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--memory-fraction", type=float, default=0.75, help="CUDA memory fraction reserved for PyTorch")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
    parser.add_argument("--gate-batch-size", type=int, default=8)
    parser.add_argument("--gate-subset-size", type=int, default=None)
    parser.add_argument("--gate-subset-epochs", type=int, default=None)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--crop-seconds", type=float, default=30.0, help="center-crop spectrogram time axis to this many seconds; default 30.0 keeps the baked length")
    parser.add_argument("--data-root", type=Path, default=None, help="processed spectrogram root; defaults to data/processed/fma_medium")
    parser.add_argument("--disable-specaugment", action="store_true", help="disable SpecAugment on the training split for robustness ablations")
    args = parser.parse_args()
    raise SystemExit(run_from_args(args))


if __name__ == "__main__":
    main()
