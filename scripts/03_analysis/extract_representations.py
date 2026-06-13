from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.representations import ExtractionConfig, extract_representations


def resolve_checkpoint(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.checkpoint is not None:
        checkpoint = args.checkpoint
        run_dir = args.run_dir
    elif args.run_dir is not None:
        run_dir = args.run_dir
        checkpoint = run_dir / "best.pt"
    else:
        raise ValueError("pass either --checkpoint or --run-dir")
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return checkpoint, run_dir


def default_output_dir(args: argparse.Namespace, run_dir: Path | None) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if run_dir is not None:
        suffix = f"representations_{args.split}"
        if args.limit_test_samples is not None and args.split == "test":
            suffix += f"_n{args.limit_test_samples}"
        return run_dir / suffix
    return REPO_ROOT / "results" / "analysis" / f"representations_{args.split}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract aligned penultimate representations for a trained deep model")
    parser.add_argument("--run-dir", type=Path, default=None, help="trained run directory containing best.pt")
    parser.add_argument("--checkpoint", type=Path, default=None, help="explicit checkpoint path")
    parser.add_argument("--model", choices=["cnn", "cnn2d", "lstm", "transformer", "mamba1", "mamba2"], default=None)
    parser.add_argument("--split", choices=["training", "validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--no-intermediates", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None, help="limit the selected split for smoke tests")
    parser.add_argument("--limit-test-samples", type=int, default=None, help="compatibility alias for test-split smoke extraction")
    args = parser.parse_args()

    checkpoint, run_dir = resolve_checkpoint(args)
    limit_samples = args.limit_samples
    if args.limit_test_samples is not None:
        if args.split != "test":
            raise ValueError("--limit-test-samples can only be used with --split test")
        limit_samples = args.limit_test_samples

    manifest = extract_representations(
        ExtractionConfig(
            checkpoint_path=checkpoint,
            output_dir=default_output_dir(args, run_dir),
            model_name=args.model,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit_samples=limit_samples,
            device=args.device,
            save_intermediates=not args.no_intermediates,
            source_run_dir=run_dir,
        )
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
