from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerifyResult:
    ok: bool
    actual_duration_s: Optional[float]
    error: Optional[str]


def _ffprobe_path() -> Optional[str]:
    return shutil.which("ffprobe")


def _via_ffprobe(path: Path) -> VerifyResult:
    cmd = [
        _ffprobe_path() or "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return VerifyResult(False, None, f"ffprobe rc={out.returncode}: {out.stderr.strip()[:200]}")
        info = json.loads(out.stdout or "{}")
        dur = info.get("format", {}).get("duration")
        if dur is None:
            return VerifyResult(False, None, "ffprobe_no_duration")
        return VerifyResult(True, float(dur), None)
    except FileNotFoundError:
        return VerifyResult(False, None, "ffprobe_missing")
    except subprocess.TimeoutExpired:
        return VerifyResult(False, None, "ffprobe_timeout")
    except Exception as exc:
        return VerifyResult(False, None, f"ffprobe_err:{exc.__class__.__name__}")


def _via_mutagen(path: Path) -> VerifyResult:
    try:
        from mutagen import File as MFile  # type: ignore
    except Exception:
        return VerifyResult(False, None, "mutagen_missing")
    try:
        m = MFile(str(path))
        if m is None or not getattr(m, "info", None):
            return VerifyResult(False, None, "mutagen_unreadable")
        return VerifyResult(True, float(m.info.length), None)
    except Exception as exc:
        return VerifyResult(False, None, f"mutagen_err:{exc.__class__.__name__}")


def verify_audio(path: Path, target_seconds: float = 30.0) -> tuple[VerifyResult, bool]:
    if _ffprobe_path() is not None:
        res = _via_ffprobe(path)
        if res.ok:
            return res, (res.actual_duration_s or 0.0) >= target_seconds
    res = _via_mutagen(path)
    if res.ok:
        return res, (res.actual_duration_s or 0.0) >= target_seconds
    return res, False
