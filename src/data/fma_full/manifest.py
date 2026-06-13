from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


FMA_DIRECT_BASE = "https://files.freemusicarchive.org/storage-freemusicarchive-org/"
DEFAULT_ZIP_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"

# the upstream FMA CSVs leave these rows with blank license fields; we keep
# this override narrow and auditable because the original MP3s carry tags.
LICENSE_OVERRIDES: dict[int, tuple[str, str]] = {
    6669: (
        "Attribution-Noncommercial-Share Alike 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/us/",
    ),
    6671: (
        "Attribution-Noncommercial-Share Alike 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/us/",
    ),
    6672: (
        "Attribution-Noncommercial-Share Alike 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/us/",
    ),
    6673: (
        "Attribution-Noncommercial-Share Alike 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/us/",
    ),
    16348: (
        "Public Domain",
        "http://creativecommons.org/licenses/publicdomain/",
    ),
    34989: (
        "Attribution-NonCommercial-ShareAlike",
        "http://creativecommons.org/licenses/by-nc-sa/3.0/",
    ),
    38666: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38668: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38669: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38670: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38671: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38672: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38673: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38674: (
        "Attribution-Noncommercial-No Derivative Works 3.0 United States",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/us/",
    ),
    38675: (
        "FMA-Limited",
        "http://freemusicarchive.org/FMA_License",
    ),
    38678: (
        "FMA-Limited",
        "http://freemusicarchive.org/FMA_License",
    ),
    38679: (
        "FMA-Limited",
        "http://freemusicarchive.org/FMA_License",
    ),
    38691: (
        "FMA-Limited",
        "http://freemusicarchive.org/FMA_License",
    ),
    38692: (
        "FMA-Limited",
        "http://freemusicarchive.org/FMA_License",
    ),
}


def _track_id_to_paths(track_id: int) -> tuple[str, str, str]:
    six = f"{track_id:06d}"
    first3 = six[:3]
    member = f"fma_full/{first3}/{six}.mp3"
    rel = f"{first3}/{six}.mp3"
    return six, first3, rel, member  # type: ignore[return-value]


def _direct_url(track_file: str | float | None) -> str | None:
    if track_file is None:
        return None
    if isinstance(track_file, float):
        return None
    s = str(track_file).strip()
    if not s or s.lower() == "nan":
        return None
    return FMA_DIRECT_BASE + s.lstrip("/")


def _ia_candidates(album_url: str | float | None, album_id, track_file: str | float | None) -> list[str]:
    out: list[str] = []
    basename = None
    if isinstance(track_file, str) and track_file.strip():
        basename = os.path.basename(track_file.strip())
    album_slug = None
    if isinstance(album_url, str) and album_url.strip():
        parts = [p for p in album_url.strip().rstrip("/").split("/") if p]
        if parts:
            album_slug = parts[-1]
    aid = None
    try:
        if album_id is not None and not (isinstance(album_id, float) and pd.isna(album_id)):
            aid = int(album_id)
    except (TypeError, ValueError):
        aid = None
    if aid is not None and basename:
        out.append(f"https://archive.org/download/fma_album_{aid}/{basename}")
    if album_slug and basename:
        out.append(f"https://archive.org/download/{album_slug}/{basename}")
    return out


def build_manifest(
    benchmark_manifest_csv: Path,
    raw_tracks_csv: Path,
    output_path: Path,
    zip_url: str = DEFAULT_ZIP_URL,
    splits: Iterable[str] | None = None,
) -> dict[str, int]:
    bench = pd.read_csv(benchmark_manifest_csv)
    if splits is not None:
        bench = bench[bench["split"].isin(list(splits))]
    needed_ids = set(int(x) for x in bench["track_id"].tolist())

    raw = pd.read_csv(raw_tracks_csv, low_memory=False)
    raw = raw[raw["track_id"].isin(needed_ids)]
    raw_by_id = {int(row["track_id"]): row for _, row in raw.iterrows()}

    rows = []
    counts = {
        "total": 0,
        "tier1": 0,
        "tier2": 0,
        "tier3": 0,
        "missing_meta": 0,
        "missing_license": 0,
        "license_overrides": 0,
    }
    for _, b in bench.iterrows():
        tid = int(b["track_id"])
        six, first3, rel, member = _track_id_to_paths(tid)
        meta = raw_by_id.get(tid)
        if meta is None:
            direct = None
            ia = []
            license_title = None
            license_url = None
            track_url = None
            counts["missing_meta"] += 1
        else:
            direct = _direct_url(meta.get("track_file"))
            ia = _ia_candidates(meta.get("album_url"), meta.get("album_id"), meta.get("track_file"))
            license_title = meta.get("license_title")
            license_url = meta.get("license_url")
            track_url = meta.get("track_url")

        license_title = _clean(license_title)
        license_url = _clean(license_url)
        if (not license_title or not license_url) and tid in LICENSE_OVERRIDES:
            override_title, override_url = LICENSE_OVERRIDES[tid]
            license_title = license_title or override_title
            license_url = license_url or override_url
            counts["license_overrides"] += 1
        if not license_title or not license_url:
            counts["missing_license"] += 1

        row = {
            "track_id": tid,
            "split": b["split"],
            "genre_top": b.get("genre_top"),
            "six_digit": six,
            "first3": first3,
            "relative_path": rel,
            "zip_member": member,
            "tier1_direct_url": direct,
            "tier2_ia_urls": ia,
            "tier3_zip_url": zip_url,
            "license_title": license_title,
            "license_url": license_url,
            "track_url": _clean(track_url),
        }
        rows.append(row)
        counts["total"] += 1
        if direct:
            counts["tier1"] += 1
        if ia:
            counts["tier2"] += 1
        counts["tier3"] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "rows": rows}, fh)
    return counts


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def load_manifest(path: Path) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return obj.get("counts", {}), obj.get("rows", [])
