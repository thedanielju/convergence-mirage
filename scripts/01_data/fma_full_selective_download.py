from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fma_full.manifest import (
    DEFAULT_ZIP_URL,
    build_manifest,
    load_manifest,
)
from src.data.fma_full.fetchers import (
    fetch_direct,
    fetch_internet_archive,
    fetch_zip_member,
    make_session,
)
from src.data.fma_full.verify import verify_audio
from src.data.fma_full.zip_byterange import ZipByteRange


DEFAULT_BENCH_MANIFEST = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready" / "manifest.csv"
DEFAULT_RAW_TRACKS = REPO_ROOT / "data" / "raw" / "fma_metadata" / "raw_tracks.csv"
DEFAULT_MANIFEST_OUT = REPO_ROOT / "data" / "fma_full" / "manifest.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "fma_full" / "raw"
DEFAULT_AUDIT_LOG = REPO_ROOT / "data" / "fma_full" / "download_audit.jsonl"
DEFAULT_CD_CACHE = REPO_ROOT / "data" / "fma_full" / ".cd_cache.bin"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditWriter:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        with self.lock:
            self.fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


def _process_track(
    row: dict,
    output_dir: Path,
    session,
    zip_client: ZipByteRange,
    target_seconds: float,
    skip_zip: bool,
) -> dict:
    tid = row["track_id"]
    six = row["six_digit"]
    first3 = row["first3"]
    out_path = output_dir / first3 / f"{six}.mp3"

    # we try three sources in order of preference and stop at the first that works:
    # the direct url, then internet archive mirrors, then a ranged read of the zip.
    chosen = None
    attempts = []
    if row.get("tier1_direct_url"):
        res = fetch_direct(row["tier1_direct_url"], out_path, session)
        attempts.append(res)
        if res.ok:
            chosen = res
    if chosen is None and row.get("tier2_ia_urls"):
        res = fetch_internet_archive(list(row["tier2_ia_urls"]), out_path, session)
        attempts.append(res)
        if res.ok:
            chosen = res
    if chosen is None and not skip_zip:
        res = fetch_zip_member(zip_client, row["zip_member"], out_path)
        attempts.append(res)
        if res.ok:
            chosen = res

    actual_duration = None
    verify_err = None
    if chosen is not None:
        vr, ok_dur = verify_audio(out_path, target_seconds=target_seconds)
        if vr.ok:
            actual_duration = vr.actual_duration_s
            if not ok_dur:
                verify_err = f"duration_below_{target_seconds}s"
        else:
            verify_err = vr.error

    record = {
        "ts_iso": _now_iso(),
        "track_id": tid,
        "split": row.get("split"),
        "tier_used": chosen.tier if chosen else None,
        "ok": chosen is not None and verify_err is None,
        "http_status": chosen.http_status if chosen else None,
        "zip_offset": chosen.zip_offset if chosen else None,
        "source_url": chosen.source_url if chosen else None,
        "bytes": chosen.bytes if chosen else 0,
        "sha256": chosen.sha256 if chosen else None,
        "declared_duration_s": float(target_seconds),
        "actual_duration_s": actual_duration,
        "license_title": row.get("license_title"),
        "license_url": row.get("license_url"),
        "track_url": row.get("track_url"),
        "out_path": str(out_path),
        "attempts": [
            {
                "tier": a.tier,
                "ok": a.ok,
                "http_status": a.http_status,
                "bytes": a.bytes,
                "error": a.error,
                "url": a.source_url,
            }
            for a in attempts
        ],
        "error": verify_err if chosen is not None else (attempts[-1].error if attempts else "no_attempts"),
    }
    return record


def _split_filter(rows: list[dict], split: str) -> list[dict]:
    if split == "all":
        return rows
    return [r for r in rows if r.get("split") == split]


def _ensure_manifest(args) -> tuple[dict, list[dict]]:
    path = Path(args.manifest_path)
    if not path.exists() or args.rebuild_manifest:
        bench = Path(args.bench_manifest)
        raw = Path(args.tracks_csv)
        if not bench.exists():
            raise SystemExit(f"benchmark manifest not found: {bench}")
        if not raw.exists():
            raise SystemExit(f"raw_tracks.csv not found: {raw}")
        build_manifest(bench, raw, path, zip_url=args.zip_url)
    return load_manifest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective FMA Full retrieval")
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_OUT))
    parser.add_argument("--bench-manifest", default=str(DEFAULT_BENCH_MANIFEST))
    parser.add_argument("--tracks-csv", default=str(DEFAULT_RAW_TRACKS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--audit-log", default=str(DEFAULT_AUDIT_LOG))
    parser.add_argument("--cd-cache", default=str(DEFAULT_CD_CACHE))
    parser.add_argument("--zip-url", default=DEFAULT_ZIP_URL)
    parser.add_argument("--split", choices=["train", "val", "validation", "test", "all"], default="test")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--build-manifest-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts, rows = _ensure_manifest(args)
    print(f"manifest counts: {counts}", flush=True)
    if args.build_manifest_only:
        return

    split_aliases = {
        "train": "training",
        "val": "validation",
    }
    split_key = split_aliases.get(args.split, args.split)
    rows = _split_filter(rows, split_key)
    if args.max_tracks is not None:
        rows = rows[: args.max_tracks]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        before = len(rows)
        rows = [r for r in rows if not (output_dir / r["first3"] / f"{r['six_digit']}.mp3").exists()]
        print(f"resume: {before} -> {len(rows)} pending", flush=True)

    if args.dry_run:
        for r in rows[:10]:
            print(json.dumps({"track_id": r["track_id"], "tier1": r.get("tier1_direct_url"), "tier2": r.get("tier2_ia_urls"), "zip_member": r["zip_member"]}))
        print(f"dry-run total pending: {len(rows)}")
        return

    audit = AuditWriter(Path(args.audit_log))
    session = make_session(pool_size=max(args.workers * 2, 16))
    zip_client = ZipByteRange(args.zip_url, cache_path=Path(args.cd_cache), session=make_session(pool_size=8))

    total_bytes = 0
    ok_count = 0
    fail_count = 0
    started = time.time()

    def runner(row):
        return _process_track(row, output_dir, session, zip_client, args.target_seconds, args.skip_zip)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(runner, r): r for r in rows}
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                audit.write(rec)
                if rec["ok"]:
                    ok_count += 1
                    total_bytes += rec["bytes"]
                else:
                    fail_count += 1
                if i % 10 == 0 or i == len(futures):
                    elapsed = time.time() - started
                    rate = (i / elapsed) if elapsed > 0 else 0.0
                    print(
                        f"progress {i}/{len(futures)} ok={ok_count} fail={fail_count} bytes={total_bytes} rate={rate:.2f}/s",
                        flush=True,
                    )
                if args.max_bytes is not None and total_bytes >= args.max_bytes:
                    print(f"max-bytes reached: {total_bytes} >= {args.max_bytes}", flush=True)
                    break
    finally:
        audit.close()

    print(
        json.dumps(
            {
                "ok": ok_count,
                "fail": fail_count,
                "bytes": total_bytes,
                "elapsed_s": time.time() - started,
                "split": split_key,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
