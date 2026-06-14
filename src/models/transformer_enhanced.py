from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.base import BaseStockModel, FeatureProjection, PredictionHead


# ---------------------------------------------------------------------------
# Positional Encoding (same as original)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (non-learnable buffer)."""

    def __init__(self, d_model: int = 64, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learned positional embedding via nn.Parameter."""

    def __init__(self, d_model: int = 64, max_len: int = 256):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Learnable weighted average over time steps."""

    def __init__(self, d_model: int = 64, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or max(d_model // 2, 16)
        self.attention_net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention_net(x).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class MeanPooling(nn.Module):
    """Simple mean over all time steps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class MaxPooling(nn.Module):
    """Max over all time steps (per feature)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values


# ---------------------------------------------------------------------------
# Deep PredictionHead (multi-hidden-layer variant)
# ---------------------------------------------------------------------------

class DeepPredictionHead(nn.Module):
    """Prediction head with *n* hidden layers (default n=2).

    Architecture: Linear -> Act -> Drop -> [Linear -> Act -> Drop] x (n-1) -> Linear(->1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError(f"num_hidden_layers >= 1, got {num_hidden_layers}")

        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.01),
        }
        act_fn = activations.get(activation)
        if act_fn is None:
            raise ValueError(f"Unknown activation '{activation}'")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i in range(num_hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"DeepPredictionHead expects [B, H], got {tuple(context.shape)}")
        return self.net(context).squeeze(-1)


# ---------------------------------------------------------------------------
# Enhanced Transformer Model — Deeper Encoder, No Separate FFN Block
# ---------------------------------------------------------------------------

class EnhancedTransformerModel(BaseStockModel):
    """Transformer Encoder with deeper stack for stock prediction.

    Enhancements over ``TransformerStockModel``:
        1. **Deeper Encoder** – ``num_encoder_layers`` defaults to **4**
           (vs 2 in baseline), increasing model capacity through more
           attention + FFN blocks inside the standard TransformerEncoder.
        2. **Deeper PredictionHead** – configurable number of hidden layers
           (default 2, up from 1 in baseline).
        3. **Additional pooling modes** – ``mean`` and ``max`` in addition
           to ``cls`` / ``last_step`` / ``attention``.
        4. **Flexible input dimension** – supports any ``num_features``
           (13, 18, 62, …).
        5. **Loss-agnostic** – compatible with huber / mse /
           pearson_ic / mse_ic losses via external Trainer.

    Data flow::

        Input  [B, T, F]
         -> FeatureProjection       [B, T, d_model]
         -> PositionalEncoding      [B, T, d_model]
         -> (CLS token if needed)   [B, T(+1), d_model]
         -> TransformerEncoder(L=4) [B, T(+1), d_model]   <-- deeper than baseline
         -> Pooling                 [B, d_model]
         -> Context LayerNorm       [B, d_model]
         -> DeepPredictionHead      [B]

    Config keys (extends base TransformerStockModel config):

        ──────────────────────────────────────
        Existing (same as baseline):
          d_model, input_dropout,
          num_encoder_layers (**default changed to 4**),
          num_heads, dim_feedforward,
          attn_dropout, ff_dropout, activation, norm_first,
          positional_encoding ("sinusoidal"|"learnable"),
          pooling ("cls"|"last_step"|"attention"|"mean"|"max"),
          cls_max_len, head_hidden_dim, head_dropout.
        ──────────────────────────────────────
        New / modified:
          deep_head               (bool, default True)
          head_num_hidden_layers  (int, default 2)    – layers in DeepPredictionHead
        ──────────────────────────────────────
    """

    _VALID_PE_TYPES = {"sinusoidal", "learnable"}
    _VALID_ACTIVATIONS = {"relu", "gelu"}
    _VALID_POOLING = {"cls", "last_step", "attention", "mean", "max"}

    def __init__(self, num_features: int = 13, config: Mapping[str, Any] | None = None):
        super().__init__(num_features=num_features, config=config)

        # ── Core dimensions ────────────────────────────────────────────
        d_model = int(self.config_value("d_model", 64))
        input_dropout = float(self.config_value("input_dropout", 0.1))

        # KEY CHANGE: default 4 encoder layers (was 2 in baseline)
        num_encoder_layers = int(self.config_value("num_encoder_layers", 4))
        num_heads = int(self.config_value("num_heads", 4))
        dim_feedforward = int(self.config_value("dim_feedforward", 128))
        attn_dropout = float(self.config_value("attn_dropout", 0.1))
        ff_dropout = float(self.config_value("ff_dropout", 0.1))
        activation = str(self.config_value("activation", "gelu"))
        norm_first = bool(self.config_value("norm_first", True))

        pos_enc_type = str(self.config_value("positional_encoding", "sinusoidal"))
        pooling = str(self.config_value("pooling", "cls")).lower()
        cls_max_len = int(self.config_value("cls_max_len", 256))

        # ── Head configs ────────────────────────────────────────────────
        deep_head = bool(self.config_value("deep_head", True))
        head_num_layers = int(self.config_value("head_num_hidden_layers", 2))
        head_hidden_dim = int(self.config_value("head_hidden_dim", 64))
        head_dropout = float(self.config_value("head_dropout", 0.3))

        # ── Validation ─────────────────────────────────────────────────
        if pos_enc_type not in self._VALID_PE_TYPES:
            raise ValueError(
                f"positional_encoding must be one of {self._VALID_PE_TYPES}, got '{pos_enc_type}'"
            )
        if activation not in self._VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {self._VALID_ACTIVATIONS}, got '{activation}'"
            )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"pooling must be one of {self._VALID_POOLING}, got '{pooling}'"
            )

        # Store for forward pass
        self.d_model = d_model
        self.pos_enc_type = pos_enc_type
        self.pooling = pooling

        # Legacy compat
        _legacy_cls = bool(self.config_value("use_cls_token", None))
        if _legacy_cls and pooling != "cls":
            self.pooling = "cls"

        # ── 1. Feature projection ──────────────────────────────────────
        self.input_proj = FeatureProjection(
            num_features=self.num_features,
            d_model=d_model,
            dropout=input_dropout,
            use_layer_norm=True,
        )

        # ── 2. Positional encoding ─────────────────────────────────────
        pe_cls = (
            SinusoidalPositionalEncoding if pos_enc_type == "sinusoidal"
            else LearnablePositionalEncoding
        )
        self.pos_encoder = pe_cls(d_model=d_model, max_len=cls_max_len)

        # ── 3. Pooling-specific modules ────────────────────────────────
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.xavier_uniform_(self.cls_token)
        elif self.pooling == "attention":
            self.attn_pool = AttentionPooling(d_model=d_model)
        elif self.pooling == "mean":
            self.mean_pool = MeanPooling()
        elif self.pooling == "max":
            self.max_pool = MaxPooling()
        # last_step: no extra module needed

        # ── 4. Deeper Transformer encoder ──────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=max(attn_dropout, ff_dropout),
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        encoder_norm = nn.LayerNorm(d_model) if norm_first else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
            norm=encoder_norm,
        )

        # ── 5. Context norm & prediction head ─────────────────────────
        self.context_norm = nn.LayerNorm(d_model)

        if deep_head and head_num_layers >= 2:
            self.head = DeepPredictionHead(
                input_dim=d_model,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
                activation=activation,
                num_hidden_layers=head_num_layers,
            )
        else:
            self.head = PredictionHead(
                input_dim=d_model,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
                activation=activation,
            )

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: [B, T, F] -> pred_score [B]."""
        self._validate_input(x)

        z = self.input_proj(x)                    # [B, T, d_model]
        z = self.pos_encoder(z)                    # [B, T, d_model]

        if self.pooling == "cls":
            cls = self.cls_token.expand(z.size(0), -1, -1)
            z = torch.cat([cls, z], dim=1)         # [B, T+1, d_model]

        encoded = self.encoder(z)                   # [B, T(+1), d_model]

        # Pooling
        if self.pooling == "cls":
            context = encoded[:, 0]
        elif self.pooling == "attention":
            context = self.attn_pool(encoded)
        elif self.pooling == "mean":
            context = self.mean_pool(encoded)
        elif self.pooling == "max":
            context = self.max_pool(encoded)
        else:  # last_step
            context = encoded[:, -1]

        context = self.context_norm(context)
        return self.head(context)                    # [B]

    # ── Input validation ──────────────────────────────────────────────

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"EnhancedTransformerModel expects torch.Tensor, got {type(x).__name__}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"EnhancedTransformerModel expects floating point tensor, got {x.dtype}"
            )
        if x.ndim != 3:
            raise ValueError(
                f"EnhancedTransformerModel expects [B, T, F], got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected feature dimension {self.num_features}, got {x.size(-1)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("Input contains NaN or Inf.")
