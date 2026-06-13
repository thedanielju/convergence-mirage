from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# make sure imports from the project root work when this script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import (
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    TARGET_SECONDS,
    build_mel_filterbank,
    expected_num_frames,
    preprocess_audio_file,
)


BENCHMARK_READY_DIR = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready"
DEFAULT_MANIFEST = BENCHMARK_READY_DIR / "manifest.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "fma_medium"


def process_one(args: tuple[int, str, str, str]) -> dict[str, object]:
    # unpack the single job description for one track.
    track_id, split, audio_path_str, output_root_str = args
    output_root = Path(output_root_str)
    audio_path = REPO_ROOT / audio_path_str
    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{track_id:06d}.npy"

    try:
        # build the mel filterbank and convert the audio file into a spectrogram.
        mel_filterbank = build_mel_filterbank()
        spectrogram = preprocess_audio_file(audio_path, mel_filterbank=mel_filterbank)

        # save the processed array and return a success record for the output manifest.
        np.save(output_path, spectrogram, allow_pickle=False)
        return {
            "track_id": track_id,
            "split": split,
            "audio_path": audio_path_str,
            "output_path": str(output_path.relative_to(REPO_ROOT)),
            "status": "ok",
            "shape": list(spectrogram.shape),
            "error": "",
        }
    except Exception as exc:
        # keep going even if one track fails, and store the error for review later.
        return {
            "track_id": track_id,
            "split": split,
            "audio_path": audio_path_str,
            "output_path": "",
            "status": "error",
            "shape": [],
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess benchmark-ready FMA medium tracks into log-mel spectrograms.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # turn each manifest row into the compact tuple that each worker will process.
    jobs = [
        (int(row.track_id), str(row.split), str(row.audio_path), str(args.output_dir))
        for row in manifest.itertuples(index=False)
    ]

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # process many tracks in parallel and print progress every 100 tracks.
        for index, result in enumerate(executor.map(process_one, jobs), start=1):
            results.append(result)
            if index % 100 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)} tracks")

    manifest_results = pd.DataFrame(results)

    # write one csv for all results and one smaller csv for failures only.
    manifest_results.to_csv(args.output_dir / "processed_manifest.csv", index=False)
    failures = manifest_results.loc[manifest_results["status"] != "ok"].copy()
    failures.to_csv(args.output_dir / "failed_tracks.csv", index=False)

    # collect the key run settings and counts in a json summary.
    summary = {
        "input_manifest": str(args.manifest.relative_to(REPO_ROOT)),
        "output_dir": str(args.output_dir.relative_to(REPO_ROOT)),
        "processed_tracks": int(len(results)),
        "failed_tracks": int(len(failures)),
        "processed_ok": int((manifest_results["status"] == "ok").sum()),
        "workers": args.workers,
        "sample_rate": SAMPLE_RATE,
        "target_seconds": TARGET_SECONDS,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "expected_frames": expected_num_frames(),
        "per_split_counts": manifest_results.loc[manifest_results["status"] == "ok", "split"].value_counts().sort_index().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
