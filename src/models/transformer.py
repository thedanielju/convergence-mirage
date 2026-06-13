from __future__ import annotations

import math

import torch
from torch import nn


# transformer encoder baseline for genre classification from mel spectrograms.
# input shape: (batch, 128 mel bins, 1292 time frames). we use this model as
# the attention reference point for the mamba comparison.
class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 16,
        input_dim: int = 128,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 16384,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)
        # the cls token is the single learned summary position that the final
        # classifier reads after the encoder has mixed information globally.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # pytorch's encoder layer keeps the implementation close to the paper
        # specification while letting torch select the best sdpa backend.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # sinusoidal positions are registered as a buffer so they move with the
        # module across cpu/gpu but are not trainable parameters.
        self.register_buffer("pe", self._build_positional_encoding(max_len, d_model), persistent=True)

    @staticmethod
    def _build_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        # standard vaswani et al. encoding: even dimensions use sine and odd
        # dimensions use cosine at geometrically spaced frequencies.
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _prepare_tokens(self, inputs: torch.Tensor) -> torch.Tensor:
        # dataset tensors are channel-first spectrograms. transformer tokens are
        # time frames, so transpose to batch x time x mel before projection.
        tokens = inputs.transpose(1, 2)
        tokens = self.input_projection(tokens)
        tokens = self.input_dropout(tokens)

        # prepend one cls token per batch item, then add the positional slice
        # matching the actual runtime sequence length.
        cls_tokens = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        if tokens.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {tokens.size(1)} exceeds positional encoding length {self.pe.size(1)}")

        tokens = tokens + self.pe[:, : tokens.size(1), :].to(dtype=tokens.dtype)
        return tokens

    def _encode(self, inputs: torch.Tensor, collect_intermediate: bool = False) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # we loop manually instead of calling self.encoder(tokens) so we can
        # capture a mid-layer cls vector and time average in one pass.
        tokens = self._prepare_tokens(inputs)
        intermediates: dict[str, torch.Tensor] = {}
        encoded = tokens
        for index, layer in enumerate(self.encoder.layers, start=1):
            encoded = layer(encoded)
            if collect_intermediate and index == 2:
                intermediates["layer_2_cls"] = encoded[:, 0, :]
                intermediates["layer_2_time_mean"] = encoded[:, 1:, :].mean(dim=1)

        cls_representation = encoded[:, 0, :]
        # we normalize the cls vector here so the representation includes
        # the transformer's full pooling choice, not just the raw layer output.
        penultimate = self.final_norm(cls_representation)
        if collect_intermediate:
            intermediates["penultimate"] = penultimate
        return penultimate, intermediates

    def get_penultimate(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate, _ = self._encode(inputs)
        return penultimate

    def get_intermediate_representations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, intermediates = self._encode(inputs, collect_intermediate=True)
        return intermediates

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate = self.get_penultimate(inputs)
        return self.head(penultimate)
