from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np


SAMPLE_RATE = 22050
TARGET_SECONDS = 30
TARGET_SAMPLES = SAMPLE_RATE * TARGET_SECONDS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 0.0
FMAX = SAMPLE_RATE / 2.0


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    # use the htk mel formula so the custom pipeline is deterministic and easy
    # to match against librosa with equivalent settings.
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    # inverse transform for placing triangular filters on fft frequency bins.
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    # build a fixed mel projection once and reuse it for all tracks. area
    # normalization keeps filters comparable across low and high frequencies.
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), num=n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    filterbank = np.zeros((n_mels, len(fft_freqs)), dtype=np.float32)
    for mel_idx in range(n_mels):
        left, center, right = hz_points[mel_idx : mel_idx + 3]

        rising = (fft_freqs >= left) & (fft_freqs <= center)
        falling = (fft_freqs >= center) & (fft_freqs <= right)

        if center > left:
            filterbank[mel_idx, rising] = (fft_freqs[rising] - left) / (center - left)
        if right > center:
            filterbank[mel_idx, falling] = (right - fft_freqs[falling]) / (right - center)

    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-12)
    filterbank *= enorm[:, np.newaxis]
    return filterbank


def decode_audio_ffmpeg(
    audio_path: Path,
    sample_rate: int = SAMPLE_RATE,
    target_samples: int = TARGET_SAMPLES,
) -> np.ndarray:
    # ffmpeg handles mp3 decoding consistently and avoids relying on librosa for
    # the main mel pipeline.
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip() or "ffmpeg decode failed"
        raise RuntimeError(error)

    waveform = np.frombuffer(result.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError("decoded waveform is empty")

    # every model sees the same 30-second window length; short clips are padded
    # and long clips are truncated.
    if waveform.size < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[: waveform.size] = waveform
        waveform = padded
    elif waveform.size > target_samples:
        waveform = waveform[:target_samples]

    return waveform


def compute_log_mel_spectrogram(
    waveform: np.ndarray,
    mel_filterbank: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    # center padding mirrors common stft behavior and gives the expected 1292
    # frames for a 30-second clip at 22,050 hz with hop length 512.
    pad = n_fft // 2
    centered_waveform = np.pad(waveform, (pad, pad), mode="constant")
    frame_count = expected_num_frames(target_samples=waveform.shape[0], n_fft=n_fft, hop_length=hop_length)
    frame_starts = np.arange(frame_count, dtype=np.int64) * hop_length
    frames = np.stack([centered_waveform[start : start + n_fft] for start in frame_starts], axis=0)
    window = np.hanning(n_fft).astype(np.float32, copy=False)
    windowed_frames = frames * window[np.newaxis, :]
    spectrum = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    power_spectrogram = (np.abs(spectrum) ** 2).T
    mel_spectrogram = mel_filterbank @ power_spectrogram

    # log compression reduces the dynamic range before per-track z-scoring.
    log_mel = np.log1p(mel_spectrogram)

    mean = float(log_mel.mean())
    std = float(log_mel.std())
    if std < 1e-8:
        std = 1.0
    # normalize each track independently; the dataset loader should not
    # re-normalize these saved tensors.
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32, copy=False)


def preprocess_audio_file(audio_path: Path, mel_filterbank: np.ndarray | None = None) -> np.ndarray:
    # callers can pass a prebuilt filterbank to avoid rebuilding it per track.
    if mel_filterbank is None:
        mel_filterbank = build_mel_filterbank()
    waveform = decode_audio_ffmpeg(audio_path)
    return compute_log_mel_spectrogram(waveform, mel_filterbank)


def expected_num_frames(
    target_samples: int = TARGET_SAMPLES,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> int:
    # with centered framing, the padded signal has target_samples + n_fft samples.
    centered_samples = target_samples + n_fft
    return 1 + math.floor((centered_samples - n_fft) / hop_length)
