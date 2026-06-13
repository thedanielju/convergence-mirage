"""Validate our custom NumPy/SciPy mel spectrogram pipeline against librosa.

This script samples a small set of tracks from the benchmark-ready manifest, runs
both the project's pipeline (``src/data/preprocessing.py``) and librosa's
reference implementation on the same decoded waveform, and reports how closely
the two agree.

We run two comparisons:

1. **Matched settings**: librosa is forced to use the same mel-scale convention
   (``htk=True``), window, padding, and post-processing (``log1p`` + per-track
   z-score) as the custom pipeline. This should produce near-identical output.
2. **Librosa defaults**: librosa is run with its out-of-the-box mel spectrogram
   settings (Slaney mel, ``power_to_db``) for reference. This will NOT match but
   is included to show how large the gap is and make the project's choices
   explicit.

Run::

    python scripts/validate_mel_spectrograms.py --num-tracks 8

Verdict thresholds on the matched-settings relative difference:

* ``< 1e-3`` : effectively identical (numerical float noise)
* ``< 1e-2`` : acceptable (minor window / padding / float ordering differences)
* ``>= 1e-2`` : investigate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.data.preprocessing import (  # noqa: E402
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    TARGET_SAMPLES,
    build_mel_filterbank,
    compute_log_mel_spectrogram,
)

try:
    import librosa
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "librosa is required for this validation script. Install with `pip install librosa`."
    ) from exc


MANIFEST_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready" / "manifest.csv"


def load_waveform(audio_path: Path) -> np.ndarray:
    """Load a 30-second mono waveform at the project's sample rate.

    Uses ``librosa.load`` (soundfile backend) so the script does not need
    ffmpeg on the PATH. The same truncate/pad logic as
    ``decode_audio_ffmpeg`` is applied so both pipelines see an identical
    input waveform.
    """
    # load audio as mono at the same sample rate used by the project pipeline.
    waveform, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    waveform = waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        raise RuntimeError(f"decoded waveform is empty for {audio_path}")
    if waveform.size < TARGET_SAMPLES:
        # pad short clips so every example has the exact same length.
        padded = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        padded[: waveform.size] = waveform
        waveform = padded
    elif waveform.size > TARGET_SAMPLES:
        # trim long clips so both pipelines receive the same 30-second window.
        waveform = waveform[:TARGET_SAMPLES]
    return waveform


def librosa_matched(waveform: np.ndarray) -> np.ndarray:
    """Run librosa with settings matched to the project pipeline.

    Matches:
    - HTK mel formula (project uses 2595*log10(1+f/700))
    - Slaney filter norm (project divides by (right - left) -> same as norm='slaney')
    - n_fft=2048, hop=512, n_mels=128, fmin=0, fmax=sr/2
    - Hann window, symmetric (numpy.hanning) via sym=True
    - center=True with constant zero padding
    - Post: log1p then per-track z-score
    """
    # force librosa to use the same symmetric hann window as the project code.
    sym_hann = np.hanning(N_FFT).astype(np.float32)
    stft = librosa.stft(
        y=waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=sym_hann,
        center=True,
        pad_mode="constant",
    )
    power = np.abs(stft) ** 2

    # build a mel filterbank with settings chosen to match the project pipeline.
    mel_fb = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        htk=True,
        norm="slaney",
    )
    mel_power = mel_fb @ power
    log_mel = np.log1p(mel_power).astype(np.float32)

    # normalize this track so the output scale matches the project output.
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    if std < 1e-8:
        std = 1.0
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32, copy=False)


def librosa_defaults(waveform: np.ndarray) -> np.ndarray:
    """Run librosa's default mel spectrogram (Slaney mel, power_to_db).

    This is included as a reference to show the gap between the project's log1p
    choice and the more common power_to_db convention. The output is also
    per-track z-scored so that the relative difference is on comparable scales.
    """
    # this version keeps librosa's usual mel-spectrogram choices for comparison.
    mel_power = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel_power, ref=1.0).astype(np.float32)

    # normalize here too so the comparison focuses on shape, not raw scale.
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    if std < 1e-8:
        std = 1.0
    normalized = (log_mel - mean) / std
    return normalized.astype(np.float32, copy=False)


def relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-invariant relative L1 difference."""
    # divide by the average size of the reference values so the score is scale-aware.
    denom = float(np.abs(b).mean())
    if denom < 1e-12:
        denom = 1.0
    return float(np.abs(a - b).mean() / denom)


def verdict(rel: float) -> str:
    if rel < 1e-3:
        return "identical (float noise)"
    if rel < 1e-2:
        return "acceptable (minor numeric differences)"
    if rel < 1e-1:
        return "noticeable - worth inspecting"
    return "large - investigate"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tracks", type=int, default=8, help="Number of tracks to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for track sampling")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(MANIFEST_PATH)
    rng = np.random.default_rng(args.seed)

    # randomly pick a small group of tracks so validation stays fast.
    sample = manifest.sample(n=min(args.num_tracks, len(manifest)), random_state=rng.integers(0, 2**31 - 1))

    # build the project's mel filterbank once and reuse it for every track.
    mel_fb = build_mel_filterbank()

    print(f"Comparing project pipeline vs librosa on {len(sample)} tracks")
    print(f"  sample_rate={SAMPLE_RATE}, n_fft={N_FFT}, hop={HOP_LENGTH}, n_mels={N_MELS}")
    print()
    print(f"{'track_id':>10}  {'matched':>12}  {'defaults':>12}  verdict (matched)")
    print("-" * 70)

    matched_diffs: list[float] = []
    default_diffs: list[float] = []

    for _, row in sample.iterrows():
        # normalize path separators so the manifest works on windows and wsl.
        rel_path = str(row["audio_path"]).replace("\\", "/")
        audio_path = REPO_ROOT / rel_path
        if not audio_path.exists():
            print(f"{row['track_id']:>10}  MISSING ({audio_path})")
            continue

        try:
            waveform = load_waveform(audio_path)
            ours = compute_log_mel_spectrogram(waveform, mel_fb)
            theirs_matched = librosa_matched(waveform)
            theirs_default = librosa_defaults(waveform)
        except Exception as exc:  # pragma: no cover
            print(f"{row['track_id']:>10}  ERROR: {exc}")
            continue

        # stop early if the two pipelines produced different output shapes.
        if ours.shape != theirs_matched.shape:
            print(
                f"{row['track_id']:>10}  SHAPE MISMATCH: ours={ours.shape} "
                f"librosa={theirs_matched.shape}"
            )
            continue

        rel_matched = relative_difference(ours, theirs_matched)
        rel_default = relative_difference(ours, theirs_default)
        matched_diffs.append(rel_matched)
        default_diffs.append(rel_default)

        # print one line per track so you can see how close the results are.
        print(
            f"{row['track_id']:>10}  {rel_matched:>12.6f}  {rel_default:>12.6f}  "
            f"{verdict(rel_matched)}"
        )

    print("-" * 70)
    if matched_diffs:
        # report average and worst-case differences across the sampled tracks.
        print(
            f"{'mean':>10}  {np.mean(matched_diffs):>12.6f}  "
            f"{np.mean(default_diffs):>12.6f}"
        )
        print(
            f"{'max':>10}  {np.max(matched_diffs):>12.6f}  "
            f"{np.max(default_diffs):>12.6f}"
        )
        print()
        overall = verdict(float(np.mean(matched_diffs)))
        print(f"Overall matched-settings verdict: {overall}")
        print()
        print("Notes:")
        print("  - 'matched' forces librosa to use the project's conventions")
        print("    (htk mel, symmetric hann, log1p, per-track z-score).")
        print("  - 'defaults' uses librosa's stock mel + power_to_db for reference.")
        print("    A large 'defaults' diff is expected and NOT a problem - it only")
        print("    shows the project picked log1p over dB scaling.")
        return 0
    print("No tracks were successfully processed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
