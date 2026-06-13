from __future__ import annotations

import binascii
import bz2
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import requests


EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CD_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"

ZIP64_EXTRA_ID = 0x0001


@dataclass
class MemberInfo:
    filename: str
    compress_method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


# we fetch only the zip central directory first because downloading the full
# archive would cost tens of gigabytes for a single track request.
class ZipByteRange:
    def __init__(self, url: str, cache_path: Optional[Path] = None, session: Optional[requests.Session] = None):
        self.url = url
        self.cache_path = cache_path
        self._session = session or requests.Session()
        self._lock = Lock()
        self._cd_bytes: Optional[bytes] = None
        self._cd_offset: Optional[int] = None
        self._index: dict[str, MemberInfo] = {}
        self._total_size: Optional[int] = None

    def _head(self) -> int:
        if self._total_size is not None:
            return self._total_size
        r = self._session.head(self.url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("server does not advertise byte ranges")
        self._total_size = int(r.headers["Content-Length"])
        return self._total_size

    def _range(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        for attempt in range(4):
            try:
                r = self._session.get(self.url, headers=headers, timeout=300, stream=False)
                if r.status_code in (200, 206):
                    return r.content
                if r.status_code == 416:
                    raise RuntimeError(f"416 range not satisfiable {start}-{end}")
            except requests.RequestException:
                if attempt == 3:
                    raise
        raise RuntimeError(f"range fetch failed {start}-{end}")

    def _ensure_cd(self) -> None:
        with self._lock:
            if self._cd_bytes is not None:
                return
            if self.cache_path is not None and self.cache_path.exists():
                try:
                    blob = self.cache_path.read_bytes()
                    cd_offset = struct.unpack("<Q", blob[:8])[0]
                    self._cd_offset = cd_offset
                    self._cd_bytes = blob[8:]
                    self._build_index()
                    return
                except Exception:
                    pass
            total = self._head()
            tail_size = min(65557 + 56, total)
            tail = self._range(total - tail_size, total - 1)
            eocd_pos = tail.rfind(EOCD_SIG)
            if eocd_pos < 0:
                raise RuntimeError("EOCD signature not found")
            (
                _disk,
                _disk_cd,
                _entries_disk,
                entries_total,
                cd_size,
                cd_offset,
                _comment_len,
            ) = struct.unpack("<HHHHIIH", tail[eocd_pos + 4 : eocd_pos + 22])
            # sentinel values indicate zip64; we must follow the locator chain
            # to find the 64-bit eocd before we can locate the central directory.
            if (
                cd_offset == 0xFFFFFFFF
                or cd_size == 0xFFFFFFFF
                or entries_total == 0xFFFF
            ):
                loc_pos = tail.rfind(EOCD64_LOCATOR_SIG, 0, eocd_pos)
                if loc_pos < 0:
                    raise RuntimeError("ZIP64 locator missing")
                _, eocd64_offset, _ = struct.unpack("<IQI", tail[loc_pos + 4 : loc_pos + 20])
                eocd64_size_hint = 56 + 1024
                fetch_start = max(0, eocd64_offset)
                fetch_end = min(total - 1, eocd64_offset + eocd64_size_hint)
                eocd64_blob = self._range(fetch_start, fetch_end)
                rel = eocd64_blob.find(EOCD64_SIG)
                if rel < 0:
                    raise RuntimeError("EOCD64 signature missing")
                eocd64 = eocd64_blob[rel:]
                cd_size = struct.unpack("<Q", eocd64[40:48])[0]
                cd_offset = struct.unpack("<Q", eocd64[48:56])[0]
            cd = bytearray()
            chunk = 32 * 1024 * 1024
            pos = cd_offset
            end = cd_offset + cd_size - 1
            while pos <= end:
                stop = min(pos + chunk - 1, end)
                cd.extend(self._range(pos, stop))
                pos = stop + 1
            self._cd_offset = cd_offset
            self._cd_bytes = bytes(cd)
            if self.cache_path is not None:
                try:
                    self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cache_path.write_bytes(struct.pack("<Q", cd_offset) + self._cd_bytes)
                except Exception:
                    pass
            self._build_index()

    def _build_index(self) -> None:
        assert self._cd_bytes is not None
        buf = self._cd_bytes
        n = len(buf)
        i = 0
        idx: dict[str, MemberInfo] = {}
        while i + 46 <= n:
            if buf[i : i + 4] != CD_SIG:
                break
            (
                _ver_made,
                _ver_need,
                _flags,
                method,
                _mtime,
                _mdate,
                crc32,
                csize,
                usize,
                fname_len,
                extra_len,
                comment_len,
                _disk_start,
                _int_attr,
                _ext_attr,
                lfh_offset,
            ) = struct.unpack("<HHHHHHIIIHHHHHII", buf[i + 4 : i + 46])
            fname = buf[i + 46 : i + 46 + fname_len].decode("utf-8", errors="replace")
            extra = buf[i + 46 + fname_len : i + 46 + fname_len + extra_len]
            csize_full = csize
            usize_full = usize
            lfh_full = lfh_offset
            j = 0
            while j + 4 <= len(extra):
                hid, hsize = struct.unpack("<HH", extra[j : j + 4])
                payload = extra[j + 4 : j + 4 + hsize]
                if hid == ZIP64_EXTRA_ID:
                    p = 0
                    if usize == 0xFFFFFFFF and p + 8 <= len(payload):
                        usize_full = struct.unpack("<Q", payload[p : p + 8])[0]
                        p += 8
                    if csize == 0xFFFFFFFF and p + 8 <= len(payload):
                        csize_full = struct.unpack("<Q", payload[p : p + 8])[0]
                        p += 8
                    if lfh_offset == 0xFFFFFFFF and p + 8 <= len(payload):
                        lfh_full = struct.unpack("<Q", payload[p : p + 8])[0]
                        p += 8
                    break
                j += 4 + hsize
            idx[fname] = MemberInfo(
                filename=fname,
                compress_method=method,
                crc32=crc32,
                compressed_size=csize_full,
                uncompressed_size=usize_full,
                local_header_offset=lfh_full,
            )
            i += 46 + fname_len + extra_len + comment_len
        self._index = idx

    def member(self, name: str) -> Optional[MemberInfo]:
        self._ensure_cd()
        return self._index.get(name)

    def extract(self, name: str) -> bytes:
        info = self.member(name)
        if info is None:
            raise KeyError(name)
        # the central directory gives us the local header offset but not the
        # data start; we must read the local header to skip its variable fields.
        lfh_blob = self._range(info.local_header_offset, info.local_header_offset + 30 - 1)
        if lfh_blob[:4] != LFH_SIG:
            raise RuntimeError("local header signature mismatch")
        fname_len, extra_len = struct.unpack("<HH", lfh_blob[26:30])
        data_start = info.local_header_offset + 30 + fname_len + extra_len
        data_end = data_start + info.compressed_size - 1
        compressed = self._range(data_start, data_end)
        if info.compress_method == 0:
            raw = compressed
        elif info.compress_method == 8:
            raw = zlib.decompress(compressed, -zlib.MAX_WBITS)
        elif info.compress_method == 12:
            raw = bz2.decompress(compressed)
        else:
            raise RuntimeError(f"unsupported method {info.compress_method}")
        if len(raw) != info.uncompressed_size:
            raise RuntimeError(
                f"size mismatch got {len(raw)} expected {info.uncompressed_size}"
            )
        if (binascii.crc32(raw) & 0xFFFFFFFF) != (info.crc32 & 0xFFFFFFFF):
            raise RuntimeError("crc32 mismatch")
        return raw
