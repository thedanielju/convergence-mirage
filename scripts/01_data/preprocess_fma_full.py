from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    build_mel_filterbank,
    compute_log_mel_spectrogram,
    decode_audio_ffmpeg,
    expected_num_frames,
)


BENCHMARK_READY_DIR = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready"
DEFAULT_MANIFEST = BENCHMARK_READY_DIR / "manifest.csv"
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "fma_full" / "raw"
SPLIT_ALIASES = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "validation": "validation",
    "test": "test",
    "all": "all",
}


class AuditWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(record, sort_keys=True) + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def target_output_dir(target_seconds: int) -> Path:
    return REPO_ROOT / "data" / "processed" / f"fma_full_{target_seconds}s"


def raw_path_for_track(input_dir: Path, track_id: int) -> Path:
    six = f"{track_id:06d}"
    return input_dir / six[:3] / f"{six}.mp3"


def load_jobs(manifest_path: Path, split: str) -> list[dict[str, Any]]:
    normalized_split = SPLIT_ALIASES[split]
    jobs: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_split = SPLIT_ALIASES.get(str(row.get("split") or ""), str(row.get("split") or ""))
            if normalized_split != "all" and row_split != normalized_split:
                continue
            jobs.append(
                {
                    "track_id": int(row["track_id"]),
                    "split": row_split,
                    "genre_top": row.get("genre_top"),
                }
            )
    return jobs


def existing_ok(path: Path, expected_shape: tuple[int, int]) -> bool:
    # we treat a file as already done only if its shape matches, so a partial or
    # stale array from an interrupted run is reprocessed instead of trusted.
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return tuple(arr.shape) == expected_shape
    except Exception:
        return False


def process_one(
    job: dict[str, Any],
    *,
    input_dir: Path,
    output_dir: Path,
    target_seconds: int,
    expected_shape: tuple[int, int],
    resume: bool,
    mel_filterbank: np.ndarray,
) -> dict[str, Any]:
    track_id = int(job["track_id"])
    split = str(job["split"])
    input_path = raw_path_for_track(input_dir, track_id)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_path = split_dir / f"{track_id:06d}.npy"

    if resume and existing_ok(output_path, expected_shape):
        return {
            "ts_iso": utc_now(),
            "track_id": track_id,
            "split": split,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "skipped_existing",
            "shape": list(expected_shape),
            "error": "",
        }

    try:
        waveform = decode_audio_ffmpeg(
            input_path,
            sample_rate=SAMPLE_RATE,
            target_samples=SAMPLE_RATE * target_seconds,
        )
        spectrogram = compute_log_mel_spectrogram(waveform, mel_filterbank=mel_filterbank)
        if tuple(spectrogram.shape) != expected_shape:
            raise RuntimeError(f"expected shape {expected_shape}, got {tuple(spectrogram.shape)}")
        np.save(output_path, spectrogram, allow_pickle=False)
        return {
            "ts_iso": utc_now(),
            "track_id": track_id,
            "split": split,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "ok",
            "shape": list(spectrogram.shape),
            "error": "",
        }
    except Exception as exc:
        return {
            "ts_iso": utc_now(),
            "track_id": track_id,
            "split": split,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "status": "error",
            "shape": [],
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess selective FMA Full MP3s into 60s/120s log-mel spectrograms.")
    parser.add_argument("--target-seconds", type=int, choices=[60, 120], required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=sorted(SPLIT_ALIASES), default="test")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-missing", action="store_true", help="skip tracks whose raw MP3 has not landed yet")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit-log", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or target_output_dir(args.target_seconds)
    audit_log = args.audit_log or (output_dir / "preprocess_audit.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    args.input_dir.mkdir(parents=True, exist_ok=True)

    jobs = load_jobs(args.manifest, args.split)
    if args.limit is not None:
        jobs = jobs[: max(0, int(args.limit))]
    missing_count = 0
    if args.skip_missing:
        before = len(jobs)
        jobs = [job for job in jobs if raw_path_for_track(args.input_dir, int(job["track_id"])).exists()]
        missing_count = before - len(jobs)
        print(f"skip-missing: {before} -> {len(jobs)} ready, {missing_count} missing", flush=True)
    expected_shape = (
        N_MELS,
        expected_num_frames(target_samples=SAMPLE_RATE * args.target_seconds, n_fft=N_FFT, hop_length=HOP_LENGTH),
    )
    mel_filterbank = build_mel_filterbank()
    audit = AuditWriter(audit_log)

    counts: dict[str, int] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    process_one,
                    job,
                    input_dir=args.input_dir,
                    output_dir=output_dir,
                    target_seconds=args.target_seconds,
                    expected_shape=expected_shape,
                    resume=args.resume,
                    mel_filterbank=mel_filterbank,
                )
                for job in jobs
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                audit.write(result)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                if index % 100 == 0 or index == len(futures):
                    print(f"processed {index}/{len(futures)} counts={counts}", flush=True)
    finally:
        audit.close()

    summary = {
        "created_at_utc": utc_now(),
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "manifest": str(args.manifest),
        "split": args.split,
        "target_seconds": args.target_seconds,
        "expected_shape": list(expected_shape),
        "workers": args.workers,
        "counts": counts,
        "missing_raw_skipped": missing_count,
        "total_jobs": len(jobs),
        "sample_rate": SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "audit_log": str(audit_log),
    }
    (output_dir / f"summary_{SPLIT_ALIASES[args.split]}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if counts.get("error", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
