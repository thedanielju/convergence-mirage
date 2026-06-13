from __future__ import annotations

import torch
from torch import nn


# bidirectional LSTM baseline for genre classification from mel spectrograms.
# input shape: (batch, 128 mel bins, 1292 time frames). we include this baseline
# because its recurrent inductive bias differs fundamentally from convolution,
# attention, and Mamba, making it a useful reference for CKA comparison.
class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 16,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.representation_dim = hidden_dim * 2

        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.Sequential(
            # attention pooling learns which time frames deserve more weight
            # and stays interpretable enough to export attention_weights.
            nn.Linear(self.representation_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )
        self.final_norm = nn.LayerNorm(self.representation_dim)
        self.head = nn.Linear(self.representation_dim, num_classes)

    def _sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        # dataset tensors are channel-first spectrograms. lstm tokens are time
        # frames, so transpose to batch x time x mel before recurrent encoding.
        tokens = inputs.transpose(1, 2)
        tokens = self.input_dropout(tokens)
        sequence, _ = self.lstm(tokens)
        return sequence

    def _pool(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # scores are normalized across time so weights sum to one per example.
        scores = self.attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.final_norm(pooled), weights

    def get_penultimate(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self._sequence(inputs)
        pooled, _ = self._pool(sequence)
        return pooled

    def get_intermediate_representations(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # we export both the pooled vector and the raw temporal summary so
        # CKA can distinguish sequence-encoder geometry from pooling geometry.
        sequence = self._sequence(inputs)
        pooled, attention_weights = self._pool(sequence)
        return {
            "sequence_mean": sequence.mean(dim=1),
            "attention_pooled": pooled,
            "attention_weights": attention_weights,
            "penultimate": pooled,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        penultimate = self.get_penultimate(inputs)
        return self.head(penultimate)


def count_lstm_parameters() -> int:
    return sum(parameter.numel() for parameter in BiLSTMClassifier().parameters())
