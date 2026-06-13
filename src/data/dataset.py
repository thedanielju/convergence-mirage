from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_READY_DIR = REPO_ROOT / "data" / "splits" / "fma_medium" / "benchmark_ready"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "fma_medium"
LABEL_MAP_PATH = REPO_ROOT / "data" / "splits" / "fma_medium" / "label_map.json"
MANIFEST_PATH = BENCHMARK_READY_DIR / "manifest.csv"


SPECTROGRAM_SAMPLE_RATE = 22050
SPECTROGRAM_HOP_LENGTH = 512
SPECTROGRAM_BAKED_SECONDS = 30.0
SPECTROGRAM_BAKED_FRAMES = 1292


def crop_seconds_to_frames(crop_seconds: float) -> int:
    return int(round(crop_seconds * SPECTROGRAM_SAMPLE_RATE / SPECTROGRAM_HOP_LENGTH))


# pytorch dataset that serves pre-computed mel spectrograms for one data split.
# spectrograms were already z-scored during preprocessing, so no re-normalization
# happens here. specaugment is applied only during training.
class FMASpectrogramDataset(Dataset):
    def __init__(
        self,
        split: str,
        apply_specaugment: bool | None = None,
        crop_seconds: float | None = None,
        processed_dir: Path | str | None = None,
    ) -> None:
        self.split = split
        # default to augmenting only the training split
        self.apply_specaugment = split == "training" if apply_specaugment is None else apply_specaugment
        self.processed_dir = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
        if not self.processed_dir.is_absolute():
            self.processed_dir = REPO_ROOT / self.processed_dir

        if crop_seconds is None:
            crop_seconds = SPECTROGRAM_BAKED_SECONDS
        try:
            crop_seconds_value = float(crop_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"crop_seconds must be numeric; got {crop_seconds!r}") from exc
        if not (crop_seconds_value > 0.0):
            raise ValueError(f"crop_seconds must be positive; got {crop_seconds_value}")
        if self.processed_dir == PROCESSED_DIR and crop_seconds_value > SPECTROGRAM_BAKED_SECONDS:
            raise ValueError(
                f"crop_seconds={crop_seconds_value} exceeds baked FMA Medium spectrogram duration "
                f"({SPECTROGRAM_BAKED_SECONDS} s); longer durations need the fma full preprocessing pipeline "
                f"(scripts/01_data/preprocess_fma_full.py)"
            )
        self.crop_seconds = crop_seconds_value
        self.crop_frames = crop_seconds_to_frames(crop_seconds_value)
        if self.crop_frames < 1:
            raise ValueError(f"crop_seconds={crop_seconds_value} resolves to <1 frames")
        if self.crop_frames > SPECTROGRAM_BAKED_FRAMES:
            self.crop_frames = SPECTROGRAM_BAKED_FRAMES
        if self.processed_dir != PROCESSED_DIR:
            self.crop_frames = crop_seconds_to_frames(crop_seconds_value)

        manifest = pd.read_csv(MANIFEST_PATH)
        # load the benchmark-ready id list for this split (already excludes the
        # 12 unreadable tracks dropped during dataset validation)
        split_ids = np.loadtxt(BENCHMARK_READY_DIR / f"{split}_ids.txt", dtype=np.int64)
        split_ids = np.atleast_1d(split_ids).astype(np.int64)

        # integer label encoding from the canonical alphabetical label map
        label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        self.label_to_id = {str(label): int(index) for label, index in label_map["label_to_id"].items()}

        # reorder manifest rows to match the id file ordering so indexing is
        # deterministic across runs
        split_manifest = manifest.loc[manifest["track_id"].isin(split_ids)].copy()
        split_manifest = split_manifest.set_index("track_id").loc[split_ids].reset_index()
        self.records = split_manifest[["track_id", "genre_top"]].to_dict(orient="records")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        track_id = int(record["track_id"])
        spectrogram_path = self.processed_dir / self.split / f"{track_id:06d}.npy"
        # memory-mapped read avoids loading the whole file into ram
        spectrogram = np.load(spectrogram_path, mmap_mode="r")
        # copy into a writable float32 tensor so specaugment can modify it
        tensor = torch.from_numpy(np.array(spectrogram, dtype=np.float32, copy=True))

        time_steps = tensor.shape[-1]
        if self.crop_frames < time_steps:
            start = (time_steps - self.crop_frames) // 2
            tensor = tensor[..., start : start + self.crop_frames].contiguous()

        if self.apply_specaugment:
            tensor = apply_specaugment(tensor)

        label_id = self.label_to_id[str(record["genre_top"])]
        return tensor, label_id


# specaugment zeros out random rectangular bands along the frequency and time
# axes, forcing the model to be robust to partial information loss. this is the
# only data augmentation applied and it runs exclusively during training.
def apply_specaugment(spectrogram: torch.Tensor) -> torch.Tensor:
    augmented = spectrogram.clone()

    # masks are applied on a copy so the underlying mmap stays untouched
    frequency_bins, time_steps = augmented.shape

    # two frequency masks, each up to 20 mel bins wide
    for _ in range(2):
        width = int(torch.randint(low=0, high=21, size=(1,)).item())
        if width > 0 and width < frequency_bins:
            start = int(torch.randint(low=0, high=frequency_bins - width + 1, size=(1,)).item())
            augmented[start : start + width, :] = 0.0

    # two time masks, each up to 50 frames wide
    for _ in range(2):
        width = int(torch.randint(low=0, high=51, size=(1,)).item())
        if width > 0 and width < time_steps:
            start = int(torch.randint(low=0, high=time_steps - width + 1, size=(1,)).item())
            augmented[:, start : start + width] = 0.0

    return augmented


# quick check that a dataloader produces the expected tensor shapes
def run_sanity_check(batch_size: int = 8) -> dict[str, list[int]]:
    dataset = FMASpectrogramDataset(split="training")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch_inputs, batch_labels = next(iter(loader))
    result = {
        "inputs_shape": list(batch_inputs.shape),
        "labels_shape": list(batch_labels.shape),
    }
    if tuple(batch_inputs.shape[1:]) != (128, 1292):
        raise ValueError(f"expected input batch shape (B, 128, 1292), found {tuple(batch_inputs.shape)}")
    if batch_labels.ndim != 1:
        raise ValueError(f"expected label batch shape (B,), found {tuple(batch_labels.shape)}")
    return result
