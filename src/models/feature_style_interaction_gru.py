from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.base import BaseStockModel, FeatureProjection, PredictionHead


class FeatureStyleInteractionGRUStockModel(BaseStockModel):
    """Alpha-only GRU with a residual style interaction head.

    The GRU backbone consumes only clean alpha features. Latest-day residual
    style/liquidity features produce bounded FiLM parameters that modulate the
    alpha context before the final prediction head.
    """

    def __init__(self, num_features: int = 18, config: Mapping[str, Any] | None = None):
        super().__init__(num_features=num_features, config=config)

        alpha_feature_count = int(self.config_value("alpha_feature_count", 13))
        if alpha_feature_count <= 0:
            raise ValueError(f"alpha_feature_count must be positive, got {alpha_feature_count}")
        if alpha_feature_count >= self.num_features:
            raise ValueError(
                "FeatureStyleInteractionGRUStockModel requires style/liquidity features after "
                f"the alpha block; got num_features={self.num_features}, "
                f"alpha_feature_count={alpha_feature_count}."
            )

        d_model = int(self.config_value("d_model", 64))
        input_dropout = float(self.config_value("input_dropout", 0.1))
        rnn_hidden_dim = int(self.config_value("rnn_hidden_dim", 128))
        rnn_num_layers = int(self.config_value("rnn_num_layers", 2))
        rnn_dropout = float(self.config_value("rnn_dropout", 0.2))
        style_hidden_dim = int(self.config_value("style_hidden_dim", 64))
        interaction_hidden_dim = int(self.config_value("interaction_hidden_dim", 128))
        head_hidden_dim = int(self.config_value("head_hidden_dim", 64))
        head_dropout = float(self.config_value("head_dropout", 0.3))
        head_activation = str(self.config_value("head_activation", "leaky_relu"))
        head_negative_slope = float(self.config_value("head_negative_slope", 0.005))

        self.alpha_feature_count = alpha_feature_count
        self.style_feature_count = self.num_features - alpha_feature_count
        self.gamma_scale = float(self.config_value("gamma_scale", 0.50))
        self.beta_scale = float(self.config_value("beta_scale", 0.20))
        self.residual_scale = float(self.config_value("residual_scale", 0.25))
        self.use_style_residual_score = bool(self.config_value("use_style_residual_score", True))

        if self.gamma_scale < 0:
            raise ValueError(f"gamma_scale must be non-negative, got {self.gamma_scale}")
        if self.beta_scale < 0:
            raise ValueError(f"beta_scale must be non-negative, got {self.beta_scale}")
        if self.residual_scale < 0:
            raise ValueError(f"residual_scale must be non-negative, got {self.residual_scale}")

        self.alpha_proj = FeatureProjection(
            num_features=self.alpha_feature_count,
            d_model=d_model,
            dropout=input_dropout,
            use_layer_norm=True,
        )
        self.alpha_gru = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.alpha_norm = nn.LayerNorm(rnn_hidden_dim)

        self.style_encoder = nn.Sequential(
            nn.Linear(self.style_feature_count, style_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(style_hidden_dim),
            nn.Dropout(head_dropout),
            nn.Linear(style_hidden_dim, style_hidden_dim),
            nn.GELU(),
        )
        self.film = nn.Sequential(
            nn.Linear(style_hidden_dim, interaction_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(interaction_hidden_dim, rnn_hidden_dim * 2),
        )
        self.interaction_norm = nn.LayerNorm(rnn_hidden_dim)
        self.head = PredictionHead(
            input_dim=rnn_hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
            negative_slope=head_negative_slope,
        )
        self.style_residual_head = PredictionHead(
            input_dim=style_hidden_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
            negative_slope=head_negative_slope,
        )

        with torch.no_grad():
            film_out = self.film[-1]
            film_out.weight.mul_(0.1)
            film_out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x_alpha = x[..., : self.alpha_feature_count]
        x_style = x[..., self.alpha_feature_count :]

        alpha_z = self.alpha_proj(x_alpha)
        _, h_n = self.alpha_gru(alpha_z)
        alpha_context = self.alpha_norm(h_n[-1])

        latest_style = x_style[:, -1, :]
        style_context = self.style_encoder(latest_style)
        gamma_raw, beta_raw = self.film(style_context).chunk(2, dim=-1)

        gamma = self.gamma_scale * torch.tanh(gamma_raw)
        beta = self.beta_scale * torch.tanh(beta_raw)
        interacted_context = self.interaction_norm(alpha_context * (1.0 + gamma) + beta)

        score = self.head(interacted_context)
        if self.use_style_residual_score and self.residual_scale > 0:
            style_residual = self.style_residual_head(style_context)
            score = score + self.residual_scale * style_residual
        return score

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"FeatureStyleInteractionGRUStockModel expects a torch.Tensor, got {type(x).__name__}"
            )
        if not x.is_floating_point():
            raise TypeError(f"FeatureStyleInteractionGRUStockModel expects a floating point tensor, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(
                f"FeatureStyleInteractionGRUStockModel expects [B, T, F], got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected feature dimension {self.num_features}, got {x.size(-1)}")
        if not torch.isfinite(x).all():
            raise ValueError("FeatureStyleInteractionGRUStockModel input contains NaN or Inf values.")
