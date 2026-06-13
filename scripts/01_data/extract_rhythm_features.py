from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.features import RHYTHM_FEATURE_NAMES
from src.data.preprocessing import SAMPLE_RATE, TARGET_SAMPLES, TARGET_SECONDS, decode_audio_ffmpeg


MANIFEST_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready" / "manifest.csv"
OUTPUT_PATH = REPO_ROOT / "data" / "raw" / "fma_metadata" / "rhythm_features.csv"
SUMMARY_PATH = REPO_ROOT / "data" / "raw" / "fma_metadata" / "rhythm_summary.json"


# we replace nan/inf with 0.0 so downstream models never see non-finite values,
# which can happen when a track is silent or has a degenerate tempo estimate.
def safe_stat(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)


def compute_summary_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "skew": 0.0, "kurtosis": 0.0}
    return {
        "mean": safe_stat(np.mean(values)),
        "std": safe_stat(np.std(values)),
        "max": safe_stat(np.max(values)),
        "skew": safe_stat(skew(values, bias=False) if values.size > 2 else 0.0),
        "kurtosis": safe_stat(kurtosis(values, fisher=True, bias=False) if values.size > 3 else 0.0),
    }


def compute_ibi_stats(beat_times: np.ndarray) -> dict[str, float]:
    intervals = np.diff(beat_times)
    if intervals.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": safe_stat(np.mean(intervals)),
        "std": safe_stat(np.std(intervals)),
        "min": safe_stat(np.min(intervals)),
        "max": safe_stat(np.max(intervals)),
        "median": safe_stat(np.median(intervals)),
    }


# we prefer the project's ffmpeg-based decoder for consistency with mel
# preprocessing, and fall back to librosa if ffmpeg is unavailable or errors.
def load_waveform(audio_path: Path) -> np.ndarray:
    try:
        return decode_audio_ffmpeg(audio_path=audio_path, sample_rate=SAMPLE_RATE, target_samples=TARGET_SAMPLES)
    except Exception:
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        waveform = waveform.astype(np.float32, copy=False)
        if waveform.size < TARGET_SAMPLES:
            padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
            padded[: waveform.size] = waveform
            return padded
        return waveform[:TARGET_SAMPLES]


# librosa.beat.beat_track may return a scalar or a 1-element array depending
# on version, so we normalize both cases to a plain float.
def tempo_to_float(tempo: object) -> float:
    tempo_array = np.asarray(tempo, dtype=np.float32).reshape(-1)
    if tempo_array.size == 0:
        return 0.0
    return safe_stat(float(tempo_array[0]))


# extract all 22 rhythm features for a single track. called in parallel by
# the process pool, where each worker decodes one 30-second clip independently.
def extract_rhythm_row(record: dict[str, object]) -> dict[str, object]:
    track_id = int(record["track_id"])
    try:
        audio_path = REPO_ROOT / str(record["audio_path"]).replace("\\", "/")
        waveform = load_waveform(audio_path)

        # onset envelope is the foundation for beat tracking and tempogram
        onset_env = librosa.onset.onset_strength(y=waveform, sr=SAMPLE_RATE)
        tempo, beat_frames = librosa.beat.beat_track(y=waveform, sr=SAMPLE_RATE, onset_envelope=onset_env)
        beat_frames = np.asarray(beat_frames, dtype=np.int64)
        beat_times = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE)
        beat_strengths = onset_env[beat_frames] if beat_frames.size > 0 else np.array([], dtype=np.float32)

        # tempogram captures tempo periodicity; the top 3 peaks indicate the
        # most prominent tempo candidates (padded to 3 if fewer are found)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=SAMPLE_RATE)
        tempogram_energy = tempogram.mean(axis=0) if tempogram.size > 0 else np.array([], dtype=np.float32)
        sorted_tempogram = np.sort(tempogram_energy)[::-1]
        top_peaks = np.pad(sorted_tempogram[:3], (0, max(0, 3 - sorted_tempogram[:3].size)), constant_values=0.0)

        onset_stats = compute_summary_stats(onset_env)
        beat_stats = compute_summary_stats(beat_strengths)
        ibi_stats = compute_ibi_stats(beat_times)
        tempogram_stats = compute_summary_stats(tempogram_energy)

        return {
            "track_id": track_id,
            "tempo_bpm": tempo_to_float(tempo),
            "beat_count": int(beat_frames.size),
            "beat_density": safe_stat(float(beat_frames.size) / TARGET_SECONDS),
            "beat_strength_mean": beat_stats["mean"],
            "beat_strength_std": beat_stats["std"],
            "beat_strength_max": beat_stats["max"],
            "onset_strength_mean": onset_stats["mean"],
            "onset_strength_std": onset_stats["std"],
            "onset_strength_max": onset_stats["max"],
            "onset_strength_skew": onset_stats["skew"],
            "onset_strength_kurtosis": onset_stats["kurtosis"],
            "ibi_mean_seconds": ibi_stats["mean"],
            "ibi_std_seconds": ibi_stats["std"],
            "ibi_min_seconds": ibi_stats["min"],
            "ibi_max_seconds": ibi_stats["max"],
            "ibi_median_seconds": ibi_stats["median"],
            "tempogram_energy_mean": tempogram_stats["mean"],
            "tempogram_energy_std": tempogram_stats["std"],
            "tempogram_energy_max": tempogram_stats["max"],
            "tempogram_peak_1": safe_stat(top_peaks[0]),
            "tempogram_peak_2": safe_stat(top_peaks[1]),
            "tempogram_peak_3": safe_stat(top_peaks[2]),
            "error": "",
        }
    except Exception as exc:
        return {"track_id": track_id, **{name: 0.0 for name in RHYTHM_FEATURE_NAMES}, "error": str(exc)}


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH)
    records = manifest[["track_id", "audio_path"]].to_dict(orient="records")
    workers = max(1, (os.cpu_count() or 1) - 2)

    rows: list[dict[str, object]] = []

    # each worker decodes and featurizes one 30-second clip independently.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(extract_rhythm_row, records), start=1):
            rows.append(result)
            if index % 500 == 0 or index == len(records):
                print(f"processed {index}/{len(records)} tracks")

    rhythm_df = pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
    failed_track_ids = rhythm_df.loc[rhythm_df["error"] != "", "track_id"].astype(int).tolist()
    rhythm_df = rhythm_df[["track_id", *RHYTHM_FEATURE_NAMES]]
    rhythm_df.to_csv(OUTPUT_PATH, index=False)

    summary = {
        "track_count": int(len(rhythm_df)),
        "feature_count": len(RHYTHM_FEATURE_NAMES),
        "feature_names": RHYTHM_FEATURE_NAMES,
        "librosa_version": librosa.__version__,
        "failed_track_ids": failed_track_ids,
        "output_path": str(OUTPUT_PATH.relative_to(REPO_ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
