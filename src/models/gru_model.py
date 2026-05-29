from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.base import BaseStockModel, FeatureProjection, PredictionHead


class GRUStockModel(BaseStockModel):
    """GRU baseline for stock sequence prediction."""

    def __init__(self, num_features: int = 62, config: Mapping[str, Any] | None = None):
        super().__init__(num_features=num_features, config=config)

        d_model = int(self.config_value("d_model", 64))
        input_dropout = float(self.config_value("input_dropout", 0.1))
        rnn_hidden_dim = int(self.config_value("rnn_hidden_dim", 128))
        rnn_num_layers = int(self.config_value("rnn_num_layers", 2))
        rnn_dropout = float(self.config_value("rnn_dropout", 0.2))
        pooling = str(self.config_value("pooling", "last_hidden"))
        head_hidden_dim = int(self.config_value("head_hidden_dim", 64))
        head_dropout = float(self.config_value("head_dropout", 0.3))
        head_activation = str(self.config_value("head_activation", "relu"))
        head_negative_slope = float(self.config_value("head_negative_slope", 0.01))

        if "bidirectional" in self.config and self.config["bidirectional"] not in (False, None):
            raise ValueError("GRU baseline is defined as unidirectional; do not enable bidirectional.")
        if pooling != "last_hidden":
            raise ValueError("GRUStockModel currently supports only pooling='last_hidden'.")

        self.d_model = d_model
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        self.pooling = pooling

        self.input_proj = FeatureProjection(
            num_features=self.num_features,
            d_model=d_model,
            dropout=input_dropout,
            use_layer_norm=True,
        )
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.context_norm = nn.LayerNorm(rnn_hidden_dim)
        self.head = PredictionHead(
            input_dim=rnn_hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
            negative_slope=head_negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        z = self.input_proj(x)
        _, h_n = self.gru(z)
        context = self.context_norm(h_n[-1])
        return self.head(context)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"GRUStockModel expects a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise TypeError(f"GRUStockModel expects a floating point tensor, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"GRUStockModel expects [B, T, F], got shape {tuple(x.shape)}")
        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected feature dimension {self.num_features}, got {x.size(-1)}")
        if not torch.isfinite(x).all():
            raise ValueError("GRUStockModel input contains NaN or Inf values.")
