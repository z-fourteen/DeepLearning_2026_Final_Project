from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.models.base import BaseStockModel, FeatureProjection, PredictionHead


class RegimeGatedGRUStockModel(BaseStockModel):
    """Two-tower GRU with cross-sectional regime gates.

    The first tower consumes clean alpha features. The second tower consumes
    residual style/liquidity features and is allowed to scale, not dominate, the
    alpha score.
    """

    def __init__(self, num_features: int = 18, config: Mapping[str, Any] | None = None):
        super().__init__(num_features=num_features, config=config)

        alpha_feature_count = int(self.config_value("alpha_feature_count", 13))
        if alpha_feature_count <= 0:
            raise ValueError(f"alpha_feature_count must be positive, got {alpha_feature_count}")
        if alpha_feature_count >= self.num_features:
            raise ValueError(
                "RegimeGatedGRUStockModel requires residual style/liquidity features after "
                f"the alpha block; got num_features={self.num_features}, "
                f"alpha_feature_count={alpha_feature_count}."
            )

        d_model = int(self.config_value("d_model", 64))
        style_d_model = int(self.config_value("style_d_model", max(16, d_model // 2)))
        input_dropout = float(self.config_value("input_dropout", 0.1))
        style_dropout = float(self.config_value("style_dropout", input_dropout))
        rnn_hidden_dim = int(self.config_value("rnn_hidden_dim", 128))
        style_hidden_dim = int(self.config_value("style_hidden_dim", max(32, rnn_hidden_dim // 2)))
        rnn_num_layers = int(self.config_value("rnn_num_layers", 2))
        rnn_dropout = float(self.config_value("rnn_dropout", 0.2))
        head_hidden_dim = int(self.config_value("head_hidden_dim", 64))
        gate_hidden_dim = int(self.config_value("gate_hidden_dim", 64))
        head_dropout = float(self.config_value("head_dropout", 0.3))
        head_activation = str(self.config_value("head_activation", "leaky_relu"))
        head_negative_slope = float(self.config_value("head_negative_slope", 0.005))
        num_regimes = int(self.config_value("num_regimes", 3))
        regime_dim = int(self.config_value("regime_dim", 32))

        self.alpha_feature_count = alpha_feature_count
        self.style_feature_count = self.num_features - alpha_feature_count
        self.num_regimes = num_regimes
        self.pos_scale_min = float(self.config_value("pos_scale_min", 0.5))
        self.pos_scale_max = float(self.config_value("pos_scale_max", 1.8))
        self.neg_scale_min = float(self.config_value("neg_scale_min", 0.2))
        self.neg_scale_max = float(self.config_value("neg_scale_max", 1.2))
        self.style_scale_max = float(self.config_value("style_scale_max", 0.35))

        if num_regimes < 1:
            raise ValueError(f"num_regimes must be positive, got {num_regimes}")
        if self.pos_scale_min >= self.pos_scale_max:
            raise ValueError("pos_scale_min must be smaller than pos_scale_max")
        if self.neg_scale_min >= self.neg_scale_max:
            raise ValueError("neg_scale_min must be smaller than neg_scale_max")
        if self.style_scale_max < 0:
            raise ValueError(f"style_scale_max must be non-negative, got {self.style_scale_max}")

        self.alpha_proj = FeatureProjection(
            num_features=self.alpha_feature_count,
            d_model=d_model,
            dropout=input_dropout,
            use_layer_norm=True,
        )
        self.style_proj = FeatureProjection(
            num_features=self.style_feature_count,
            d_model=style_d_model,
            dropout=style_dropout,
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
        self.style_gru = nn.GRU(
            input_size=style_d_model,
            hidden_size=style_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.alpha_norm = nn.LayerNorm(rnn_hidden_dim)
        self.style_norm = nn.LayerNorm(style_hidden_dim)

        regime_input_dim = self.style_feature_count * 2
        self.regime_state = nn.Sequential(
            nn.Linear(regime_input_dim, regime_dim),
            nn.GELU(),
            nn.LayerNorm(regime_dim),
        )
        self.regime_router = nn.Linear(regime_dim, num_regimes)
        self.alpha_experts = nn.ModuleList(
            [
                PredictionHead(
                    input_dim=rnn_hidden_dim,
                    hidden_dim=head_hidden_dim,
                    dropout=head_dropout,
                    activation=head_activation,
                    negative_slope=head_negative_slope,
                )
                for _ in range(num_regimes)
            ]
        )

        style_context_dim = rnn_hidden_dim + style_hidden_dim
        gate_context_dim = style_context_dim + regime_dim
        self.style_adjust_head = PredictionHead(
            input_dim=style_context_dim,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
            negative_slope=head_negative_slope,
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_context_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(gate_hidden_dim, 3),
        )
        style_gate_init_bias = float(self.config_value("style_gate_init_bias", -2.0))
        with torch.no_grad():
            self.gate[-1].bias[2].fill_(style_gate_init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        x_alpha = x[..., : self.alpha_feature_count]
        x_style = x[..., self.alpha_feature_count :]

        alpha_z = self.alpha_proj(x_alpha)
        style_z = self.style_proj(x_style)
        _, alpha_h_n = self.alpha_gru(alpha_z)
        _, style_h_n = self.style_gru(style_z)
        alpha_context = self.alpha_norm(alpha_h_n[-1])
        style_context = self.style_norm(style_h_n[-1])

        regime_context = self._batch_regime_context(x_style)
        regime_weights = torch.softmax(self.regime_router(regime_context), dim=-1)
        expert_scores = torch.stack([expert(alpha_context) for expert in self.alpha_experts], dim=-1)
        alpha_score = torch.sum(expert_scores * regime_weights, dim=-1)

        style_context_pair = torch.cat([alpha_context, style_context], dim=-1)
        style_adjust = self.style_adjust_head(style_context_pair)
        gate_input = torch.cat([style_context_pair, regime_context], dim=-1)
        gate_logits = self.gate(gate_input)

        pos_scale = self._bounded_sigmoid(gate_logits[:, 0], self.pos_scale_min, self.pos_scale_max)
        neg_scale = self._bounded_sigmoid(gate_logits[:, 1], self.neg_scale_min, self.neg_scale_max)
        style_scale = self.style_scale_max * torch.sigmoid(gate_logits[:, 2])

        return (
            pos_scale * F.relu(alpha_score)
            - neg_scale * F.relu(-alpha_score)
            + style_scale * style_adjust
        )

    def _batch_regime_context(self, x_style: torch.Tensor) -> torch.Tensor:
        latest_style = x_style[:, -1, :]
        style_mean = latest_style.mean(dim=0, keepdim=True)
        style_std = latest_style.std(dim=0, keepdim=True, unbiased=False)
        batch_state = torch.cat([style_mean, style_std], dim=-1)
        regime_context = self.regime_state(batch_state)
        return regime_context.expand(x_style.size(0), -1)

    @staticmethod
    def _bounded_sigmoid(logit: torch.Tensor, low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.sigmoid(logit)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"RegimeGatedGRUStockModel expects a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise TypeError(f"RegimeGatedGRUStockModel expects a floating point tensor, got {x.dtype}")
        if x.ndim != 3:
            raise ValueError(f"RegimeGatedGRUStockModel expects [B, T, F], got shape {tuple(x.shape)}")
        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected feature dimension {self.num_features}, got {x.size(-1)}")
        if not torch.isfinite(x).all():
            raise ValueError("RegimeGatedGRUStockModel input contains NaN or Inf values.")
