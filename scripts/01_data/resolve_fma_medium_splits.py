from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# define the important input and output folders once so the rest of the script stays simple.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
TRACKS_CSV = RAW_DIR / "fma_metadata" / "tracks.csv"
AUDIO_DIR = RAW_DIR / "fma_medium"
OUTPUT_DIR = REPO_ROOT / "data" / "splits" / "fma_medium"


def track_audio_path(track_id: int) -> Path:
    # fma stores each audio file inside a folder based on the first three digits.
    track_str = f"{track_id:06d}"
    return AUDIO_DIR / track_str[:3] / f"{track_str}.mp3"


def main() -> None:
    # tracks.csv has multi-level column names, so read both header rows.
    tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)
    tracks.index = tracks.index.astype(int)

    # keep only the medium subset, then pull out the columns we want in a flat table.
    medium = tracks[tracks[("set", "subset")] == "medium"].copy()
    medium["track_id"] = medium.index
    medium["split"] = medium[("set", "split")]
    medium["artist_id"] = medium[("artist", "id")]
    medium["genre_top"] = medium[("track", "genre_top")]
    medium["audio_path"] = medium["track_id"].map(lambda track_id: str(track_audio_path(track_id).relative_to(REPO_ROOT)))
    medium["audio_exists"] = medium["track_id"].map(lambda track_id: track_audio_path(track_id).is_file())

    split_order = ["training", "validation", "test"]
    split_counts = {
        split: int((medium["split"] == split).sum())
        for split in split_order
    }

    # compare artist ids across splits to check whether the splits overlap by artist.
    artist_sets = {
        split: set(medium.loc[medium["split"] == split, "artist_id"].dropna().astype(int))
        for split in split_order
    }
    artist_overlap = {
        "training_validation": len(artist_sets["training"] & artist_sets["validation"]),
        "training_test": len(artist_sets["training"] & artist_sets["test"]),
        "validation_test": len(artist_sets["validation"] & artist_sets["test"]),
    }

    missing_audio = medium.loc[~medium["audio_exists"], ["track_id", "split", "artist_id", "genre_top", "audio_path"]].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # write the main manifest in a flat, beginner-friendly format.
    manifest = pd.DataFrame(
        {
            "track_id": medium["track_id"].astype(int),
            "split": medium["split"].astype(str),
            "artist_id": medium["artist_id"],
            "genre_top": medium["genre_top"],
            "audio_path": medium["audio_path"].astype(str),
            "audio_exists": medium["audio_exists"].astype(bool),
        }
    )
    manifest = manifest.reset_index(drop=True)
    manifest.sort_values(["split", "track_id"]).to_csv(OUTPUT_DIR / "manifest.csv", index=False)

    # also write plain text id lists for each split.
    for split in split_order:
        split_ids = medium.loc[medium["split"] == split, "track_id"].sort_values()
        (OUTPUT_DIR / f"{split}_ids.txt").write_text(
            "\n".join(str(track_id) for track_id in split_ids) + "\n",
            encoding="ascii",
        )

    if missing_audio.empty:
        (OUTPUT_DIR / "missing_audio.csv").write_text("track_id,split,artist_id,genre_top,audio_path\n", encoding="ascii")
    else:
        missing_audio.to_csv(OUTPUT_DIR / "missing_audio.csv", index=False)

    # save a json summary so later scripts can reuse these counts and paths.
    summary = {
        "tracks_csv": str(TRACKS_CSV.relative_to(REPO_ROOT)),
        "audio_dir": str(AUDIO_DIR.relative_to(REPO_ROOT)),
        "subset_filter": {"set.subset": "medium"},
        "split_source": "tracks.csv (set.split)",
        "counts": {
            "training": split_counts["training"],
            "validation": split_counts["validation"],
            "test": split_counts["test"],
            "total": int(len(medium)),
        },
        "artist_overlap": artist_overlap,
        "all_audio_present": bool(medium["audio_exists"].all()),
        "missing_audio_count": int((~medium["audio_exists"]).sum()),
        "artifacts": {
            "manifest": str((OUTPUT_DIR / "manifest.csv").relative_to(REPO_ROOT)),
            "training_ids": str((OUTPUT_DIR / "training_ids.txt").relative_to(REPO_ROOT)),
            "validation_ids": str((OUTPUT_DIR / "validation_ids.txt").relative_to(REPO_ROOT)),
            "test_ids": str((OUTPUT_DIR / "test_ids.txt").relative_to(REPO_ROOT)),
            "missing_audio": str((OUTPUT_DIR / "missing_audio.csv").relative_to(REPO_ROOT)),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Resolved FMA medium benchmark splits from tracks.csv")
    print(f"tracks.csv: {TRACKS_CSV}")
    print(f"audio dir:   {AUDIO_DIR}")
    print()
    print("Counts")
    print(f"  training:   {split_counts['training']}")
    print(f"  validation: {split_counts['validation']}")
    print(f"  test:       {split_counts['test']}")
    print(f"  total:      {len(medium)}")
    print()
    print("Artist overlap")
    for pair, overlap in artist_overlap.items():
        print(f"  {pair}: {overlap}")
    print()
    print(f"All benchmark audio present: {medium['audio_exists'].all()}")
    print(f"Missing benchmark audio:     {(~medium['audio_exists']).sum()}")
    print()
    print(f"Artifacts written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
