from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """1D convolutional feature block for temporal mel-spectrogram encoding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Conv2DBlock(nn.Module):
    """Reusable 2D time-frequency block for the CNN2D baseline.

    The 2D model treats a mel spectrogram like an image whose axes have meaning:
    frequency on one axis and time on the other.  Pooling is intentionally
    configurable because early layers can aggressively shrink the long time axis
    while the final block preserves enough spatial structure for global pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 3),
        pool_size: tuple[int, int] | None = (2, 2),
    ) -> None:
        super().__init__()
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=pool_size))
        self.block = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class CNN1D(nn.Module):
    """1D CNN baseline over mel bins-as-channels and time as the sequence axis."""

    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(128, 128, kernel_size=7),
            ConvBlock(128, 256, kernel_size=5),
            ConvBlock(256, 512, kernel_size=5),
            ConvBlock(512, 512, kernel_size=5),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(512, num_classes)

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def get_penultimate(self, inputs: torch.Tensor) -> torch.Tensor:
        # we keep this method separate from forward() so the classifier head
        # does not leak into representation analysis for CKA comparison.
        encoded = self._encode(inputs)
        pooled = self.pool(encoded).squeeze(-1)
        return pooled

    def get_intermediate_representations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        activations: dict[str, torch.Tensor] = {}
        hidden = inputs
        for index, block in enumerate(self.encoder, start=1):
            hidden = block(hidden)
            if index == 2:
                activations["block_2_mean"] = hidden.mean(dim=-1)
        penultimate = self.pool(hidden).squeeze(-1)
        activations["penultimate"] = penultimate
        return activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate = self.get_penultimate(inputs)
        return self.head(penultimate)


class CNN2D(nn.Module):
    """2D CNN baseline that preserves local time-frequency structure."""

    def __init__(self, num_classes: int = 16) -> None:
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                Conv2DBlock(1, 32, kernel_size=(5, 7), pool_size=(2, 4)),
                Conv2DBlock(32, 128, kernel_size=(3, 5), pool_size=(2, 4)),
                Conv2DBlock(128, 256, kernel_size=(3, 5), pool_size=(2, 2)),
                Conv2DBlock(256, 512, kernel_size=(3, 3), pool_size=(2, 2)),
                Conv2DBlock(512, 512, kernel_size=(3, 3), pool_size=None),
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, num_classes)

    def _prepare_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        # training scripts pass channel-free (batch, mel, time) tensors;
        # conv2d expects an explicit channel dim, so we insert one here.
        if inputs.ndim == 3:
            return inputs.unsqueeze(1)
        if inputs.ndim == 4 and inputs.shape[1] == 1:
            return inputs
        raise ValueError(f"expected input shape (B, 128, 1292) or (B, 1, 128, 1292), found {tuple(inputs.shape)}")

    def _encode(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self._prepare_inputs(inputs)
        for block in self.encoder:
            hidden = block(hidden)
        return hidden

    def get_penultimate(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(inputs)
        pooled = self.pool(encoded).flatten(1)
        return pooled

    def get_intermediate_representations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # intermediate block means let analysis scripts check whether
        # convergence appears before the final pooled representation.
        activations: dict[str, torch.Tensor] = {}
        hidden = self._prepare_inputs(inputs)
        for index, block in enumerate(self.encoder, start=1):
            hidden = block(hidden)
            if index in {2, 4}:
                activations[f"block_{index}_mean"] = hidden.mean(dim=(-2, -1))
        penultimate = self.pool(hidden).flatten(1)
        activations["penultimate"] = penultimate
        return activations

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate = self.get_penultimate(inputs)
        return self.head(penultimate)
