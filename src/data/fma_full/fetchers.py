from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .zip_byterange import ZipByteRange


@dataclass
class FetchResult:
    tier: int
    ok: bool
    http_status: Optional[int]
    bytes: int
    sha256: Optional[str]
    error: Optional[str]
    source_url: Optional[str] = None
    zip_offset: Optional[int] = None


def make_session(pool_size: int = 32) -> requests.Session:
    # we set raise_on_status=False so the caller can inspect the status code
    # and decide whether to fall through to the next tier.
    s = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "music-classification-fma-full/0.1"})
    return s


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def fetch_direct(url: str, out_path: Path, session: requests.Session, min_bytes: int = 1024) -> FetchResult:
    try:
        r = session.get(url, timeout=300, stream=True, allow_redirects=True)
        status = r.status_code
        if status != 200:
            return FetchResult(1, False, status, 0, None, f"http {status}", url)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "html" in ctype:
            return FetchResult(1, False, status, 0, None, f"html response", url)
        chunks = []
        total = 0
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                chunks.append(chunk)
                total += len(chunk)
        data = b"".join(chunks)
        if len(data) < min_bytes:
            return FetchResult(1, False, status, len(data), None, "too_small", url)
        # we check the magic bytes because some servers return a 200 html error
        # page instead of refusing with a 4xx status.
        if not (data[:3] == b"ID3" or data[:2] == b"\xff\xfb" or data[:2] == b"\xff\xf3" or data[:2] == b"\xff\xfa"):
            return FetchResult(1, False, status, len(data), None, "not_mp3", url)
        _atomic_write(out_path, data)
        return FetchResult(1, True, status, len(data), _sha256_bytes(data), None, url)
    except requests.RequestException as exc:
        return FetchResult(1, False, None, 0, None, f"req:{exc.__class__.__name__}", url)
    except Exception as exc:
        return FetchResult(1, False, None, 0, None, f"err:{exc.__class__.__name__}:{exc}", url)


def fetch_internet_archive(urls: list[str], out_path: Path, session: requests.Session, min_bytes: int = 1024) -> FetchResult:
    last_status = None
    last_err = "no_candidates"
    last_url = None
    for url in urls:
        last_url = url
        try:
            r = session.get(url, timeout=300, stream=True, allow_redirects=True)
            last_status = r.status_code
            if r.status_code != 200:
                last_err = f"http {r.status_code}"
                r.close()
                continue
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "html" in ctype:
                last_err = "html_response"
                r.close()
                continue
            data = b""
            buf = []
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    buf.append(chunk)
            data = b"".join(buf)
            if len(data) < min_bytes:
                last_err = "too_small"
                continue
            if not (data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xfa")):
                last_err = "not_mp3"
                continue
            _atomic_write(out_path, data)
            return FetchResult(2, True, r.status_code, len(data), _sha256_bytes(data), None, url)
        except requests.RequestException as exc:
            last_err = f"req:{exc.__class__.__name__}"
        except Exception as exc:
            last_err = f"err:{exc.__class__.__name__}:{exc}"
    return FetchResult(2, False, last_status, 0, None, last_err, last_url)


def fetch_zip_member(zip_client: ZipByteRange, member: str, out_path: Path) -> FetchResult:
    # tier 3 fallback: we extract a single member by byte range so we never
    # download the full archive just to get one mp3.
    try:
        info = zip_client.member(member)
        if info is None:
            return FetchResult(3, False, None, 0, None, "member_not_found", zip_client.url)
        data = zip_client.extract(member)
        _atomic_write(out_path, data)
        return FetchResult(
            3,
            True,
            206,
            len(data),
            _sha256_bytes(data),
            None,
            zip_client.url,
            zip_offset=info.local_header_offset,
        )
    except Exception as exc:
        return FetchResult(3, False, None, 0, None, f"zip:{exc.__class__.__name__}:{exc}", zip_client.url)
