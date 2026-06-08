from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class FeatureProjection(nn.Module):
    """Project raw sequence features into the model hidden dimension."""

    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.num_features = int(num_features)
        self.d_model = int(d_model)
        self.proj = nn.Linear(self.num_features, self.d_model)
        self.norm = nn.LayerNorm(self.d_model) if use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FeatureProjection expects [B, T, F], got shape {tuple(x.shape)}")
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected feature dimension {self.num_features}, got {x.size(-1)}"
            )
        return self.dropout(self.norm(self.proj(x)))


class PredictionHead(nn.Module):
    """Map an encoded stock context vector to a scalar prediction score."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "relu",
        negative_slope: float = 0.01,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if negative_slope <= 0.0:
            raise ValueError(f"negative_slope must be positive, got {negative_slope}")

        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(negative_slope=float(negative_slope)),
        }
        if activation not in activations:
            choices = ", ".join(sorted(activations))
            raise ValueError(f"Unknown activation '{activation}'. Expected one of: {choices}.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            activations[activation],
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"PredictionHead expects [B, H], got shape {tuple(context.shape)}")
        if context.size(-1) != self.input_dim:
            raise ValueError(f"Expected context dimension {self.input_dim}, got {context.size(-1)}")
        return self.net(context).squeeze(-1)


class BaseStockModel(nn.Module, ABC):
    """Base interface for sequence stock prediction models."""

    def __init__(self, num_features: int = 62, config: Mapping[str, Any] | None = None):
        super().__init__()
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.num_features = int(num_features)
        self.config = dict(config or {})

    def config_value(self, name: str, default: Any) -> Any:
        return self.config.get(name, default)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return pred_score with shape [B] for input x with shape [B, T, F]."""
