from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLIT_DIR = REPO_ROOT / "data" / "splits" / "fma_medium"
OUTPUT_DIR = SPLIT_DIR / "benchmark_ready"

TRACKS_CSV = RAW_DIR / "fma_metadata" / "tracks.csv"
FEATURES_CSV = RAW_DIR / "fma_metadata" / "features.csv"
MANIFEST_CSV = SPLIT_DIR / "manifest.csv"
SUMMARY_JSON = SPLIT_DIR / "summary.json"


def run_ffprobe(audio_path: Path) -> tuple[bool, str | None, float | None]:
    # ask ffprobe for the audio duration so we can confirm the file is readable.
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip() or "ffprobe failed", None
    duration_text = (result.stdout or "").strip().splitlines()
    if not duration_text:
        return False, "duration missing from ffprobe output", None
    try:
        duration = float(duration_text[0])
    except ValueError:
        return False, f"unparseable duration: {duration_text[0]!r}", None
    return True, None, duration


def main() -> None:
    if not MANIFEST_CSV.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {MANIFEST_CSV}. Run resolve_fma_medium_splits.py first."
        )

    # load the split manifest created by the previous script.
    manifest = pd.read_csv(MANIFEST_CSV)
    split_summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8"))

    # read the medium subset metadata from the original fma tables.
    tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)
    tracks.index = pd.to_numeric(tracks.index, errors="coerce")
    tracks = tracks[tracks.index.notna()].copy()
    tracks.index = tracks.index.astype(int)
    medium = tracks[tracks[("set", "subset")] == "medium"].copy()

    features = pd.read_csv(FEATURES_CSV, header=[0, 1, 2], index_col=0)
    features.index = pd.to_numeric(features.index, errors="coerce")
    features = features[features.index.notna()].copy()
    features.index = features.index.astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # check whether every benchmark track also has a row in features.csv.
    benchmark_ids = set(manifest["track_id"].astype(int))
    feature_ids = set(features.index)
    missing_feature_ids = sorted(benchmark_ids - feature_ids)

    # measure artist overlap again using the manifest that will feed the benchmark.
    artist_sets = {
        split: set(
            manifest.loc[manifest["split"] == split, "artist_id"].dropna().astype(int)
        )
        for split in ["training", "validation", "test"]
    }
    artist_overlap = {
        "training_validation": len(artist_sets["training"] & artist_sets["validation"]),
        "training_test": len(artist_sets["training"] & artist_sets["test"]),
        "validation_test": len(artist_sets["validation"] & artist_sets["test"]),
    }

    # count the target genres to understand class balance.
    genre_distribution = (
        manifest["genre_top"]
        .fillna("UNKNOWN")
        .value_counts()
        .rename_axis("genre_top")
        .reset_index(name="count")
    )
    genre_distribution["proportion"] = genre_distribution["count"] / len(manifest)
    genre_distribution.to_csv(OUTPUT_DIR / "genre_distribution.csv", index=False)

    majority_row = genre_distribution.iloc[0]
    baselines = {
        "random_guess_accuracy": 1.0 / manifest["genre_top"].nunique(),
        "majority_class_genre": str(majority_row["genre_top"]),
        "majority_class_accuracy": float(majority_row["count"] / len(manifest)),
    }

    # test every audio file in parallel and collect whether ffprobe can read it.
    audio_paths = [REPO_ROOT / rel_path for rel_path in manifest["audio_path"]]
    with ThreadPoolExecutor(max_workers=8) as executor:
        audio_checks = list(executor.map(run_ffprobe, audio_paths))

    # combine the original manifest columns with the readability results.
    readability = pd.DataFrame(
        {
            "track_id": manifest["track_id"].astype(int),
            "split": manifest["split"],
            "artist_id": manifest["artist_id"],
            "genre_top": manifest["genre_top"],
            "audio_path": manifest["audio_path"],
            "readable": [item[0] for item in audio_checks],
            "error": [item[1] for item in audio_checks],
            "duration_seconds": [item[2] for item in audio_checks],
        }
    )
    unreadable = readability.loc[~readability["readable"]].copy()
    unreadable.to_csv(OUTPUT_DIR / "unreadable_audio.csv", index=False)

    # keep only readable tracks for the final benchmark-ready manifest.
    benchmark_ready = readability.loc[readability["readable"]].copy()
    benchmark_ready_manifest = benchmark_ready[
        ["track_id", "split", "artist_id", "genre_top", "audio_path", "duration_seconds"]
    ].sort_values(["split", "track_id"])
    benchmark_ready_manifest.to_csv(OUTPUT_DIR / "manifest.csv", index=False)
    benchmark_ready_counts = {
        split: int((benchmark_ready["split"] == split).sum())
        for split in ["training", "validation", "test"]
    }
    for split in ["training", "validation", "test"]:
        split_ids = benchmark_ready.loc[benchmark_ready["split"] == split, "track_id"].sort_values()
        (OUTPUT_DIR / f"{split}_ids.txt").write_text(
            "\n".join(str(track_id) for track_id in split_ids) + "\n",
            encoding="ascii",
        )

    # summarize the durations of the readable audio files.
    readable_durations = readability.loc[readability["readable"], "duration_seconds"]
    duration_summary = {
        "count": int(readable_durations.shape[0]),
        "min_seconds": float(readable_durations.min()) if not readable_durations.empty else math.nan,
        "max_seconds": float(readable_durations.max()) if not readable_durations.empty else math.nan,
        "mean_seconds": float(readable_durations.mean()) if not readable_durations.empty else math.nan,
    }

    # write one summary json that describes all validation checks and outputs.
    summary = {
        "step": "2.3",
        "benchmark_subset": {
            "filter": {"set.subset": "medium"},
            "rows": int(len(medium)),
            "split_counts": split_summary["counts"],
        },
        "artist_overlap": artist_overlap,
        "genre_distribution_path": str((OUTPUT_DIR / "genre_distribution.csv").relative_to(REPO_ROOT)),
        "baselines": baselines,
        "audio_validation": {
            "method": "ffprobe duration parse",
            "benchmark_tracks_checked": int(len(readability)),
            "readable_count": int(readability["readable"].sum()),
            "unreadable_count": int((~readability["readable"]).sum()),
            "unreadable_path": str((OUTPUT_DIR / "unreadable_audio.csv").relative_to(REPO_ROOT)),
            "benchmark_ready_manifest": str((OUTPUT_DIR / "manifest.csv").relative_to(REPO_ROOT)),
            "benchmark_ready_split_counts": benchmark_ready_counts,
            "duration_summary_seconds": duration_summary,
        },
        "feature_validation": {
            "feature_rows_total": int(len(features)),
            "benchmark_ids_covered": int(len(benchmark_ids & feature_ids)),
            "missing_feature_count": int(len(missing_feature_ids)),
            "missing_feature_ids": missing_feature_ids,
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Validated FMA medium benchmark subset")
    print()
    print("Benchmark counts")
    print(f"  rows in tracks.csv medium subset: {len(medium)}")
    print(f"  training:   {split_summary['counts']['training']}")
    print(f"  validation: {split_summary['counts']['validation']}")
    print(f"  test:       {split_summary['counts']['test']}")
    print()
    print("Artist overlap")
    for pair, overlap in artist_overlap.items():
        print(f"  {pair}: {overlap}")
    print()
    print("Trivial baselines")
    print(f"  random guess accuracy:   {baselines['random_guess_accuracy']:.6f}")
    print(f"  majority class genre:    {baselines['majority_class_genre']}")
    print(f"  majority class accuracy: {baselines['majority_class_accuracy']:.6f}")
    print()
    print("Audio readability")
    print(f"  benchmark tracks checked: {len(readability)}")
    print(f"  readable:                 {int(readability['readable'].sum())}")
    print(f"  unreadable:               {int((~readability['readable']).sum())}")
    print()
    print("Feature coverage")
    print(f"  benchmark ids covered: {len(benchmark_ids & feature_ids)}")
    print(f"  missing benchmark ids: {len(missing_feature_ids)}")
    print()
    print(f"Artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
