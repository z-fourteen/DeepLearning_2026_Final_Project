from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from src.models.base import BaseStockModel, FeatureProjection, PredictionHead


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding, registered as buffer (non-learnable).

    Follows the formulation in "Attention Is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int = 64, max_len: int = 256):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.d_model = d_model
        # pe: [1, max_len, d_model]
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )                                                                            # [d_model//2]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input sequence.

        Args:
            x: FloatTensor of shape [B, T, D]

        Returns:
            FloatTensor of shape [B, T, D] with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learned positional embedding via nn.Parameter.

    Each position gets an independent vector optimized during training.
    """

    def __init__(self, d_model: int = 64, max_len: int = 256):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")

        self.d_model = d_model
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to input sequence.

        Args:
            x: FloatTensor of shape [B, T, D]

        Returns:
            FloatTensor of shape [B, T, D] with positional encoding added.
        """
        return x + self.pe[:, : x.size(1)]


class TransformerStockModel(BaseStockModel):
    """Transformer Encoder for stock sequence prediction.

    Architecture:
        Input [B, T, F]
          -> FeatureProjection     [B, T, d_model=64]
          -> PositionalEncoding    [B, T, 64]
          -> (optional CLS token)  [B, T(+1), 64]
          -> TransformerEncoder(L=2, H=4, d_ff=128, Pre-LN)  [B, T(+1), 64]
          -> Pooling               [B, 64]
          -> PredictionHead(GELU)  [B]

    Config keys:
        d_model, input_dropout,
        num_encoder_layers, num_heads, dim_feedforward,
        attn_dropout, ff_dropout, activation, norm_first,
        positional_encoding ("sinusoidal" | "learnable"),
        use_cls_token, cls_max_len,
        head_hidden_dim, head_dropout.
    """

    _VALID_PE_TYPES = {"sinusoidal", "learnable"}
    _VALID_ACTIVATIONS = {"relu", "gelu"}

    def __init__(self, num_features: int = 62, config: Mapping[str, Any] | None = None):
        super().__init__(num_features=num_features, config=config)

        d_model = int(self.config_value("d_model", 64))
        input_dropout = float(self.config_value("input_dropout", 0.1))

        num_encoder_layers = int(self.config_value("num_encoder_layers", 2))
        num_heads = int(self.config_value("num_heads", 4))
        dim_feedforward = int(self.config_value("dim_feedforward", 128))
        attn_dropout = float(self.config_value("attn_dropout", 0.1))
        ff_dropout = float(self.config_value("ff_dropout", 0.1))
        activation = str(self.config_value("activation", "gelu"))
        norm_first = bool(self.config_value("norm_first", True))

        pos_enc_type = str(self.config_value("positional_encoding", "sinusoidal"))
        use_cls_token = bool(self.config_value("use_cls_token", True))
        cls_max_len = int(self.config_value("cls_max_len", 256))

        head_hidden_dim = int(self.config_value("head_hidden_dim", 64))
        head_dropout = float(self.config_value("head_dropout", 0.3))

        # --- validation ---
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

        self.d_model = d_model
        self.pos_enc_type = pos_enc_type
        self.use_cls_token = use_cls_token
        self.pooling = "cls" if use_cls_token else "last_step"

        # 1. Feature projection
        self.input_proj = FeatureProjection(
            num_features=self.num_features,
            d_model=d_model,
            dropout=input_dropout,
            use_layer_norm=True,
        )

        # 2. Positional encoding
        pe_cls = (
            SinusoidalPositionalEncoding if pos_enc_type == "sinusoidal"
            else LearnablePositionalEncoding
        )
        self.pos_encoder = pe_cls(d_model=d_model, max_len=cls_max_len)

        # 3. Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.xavier_uniform_(self.cls_token)

        # 4. Transformer encoder
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

        # 5. Context normalization & prediction head
        self.context_norm = nn.LayerNorm(d_model)
        self.head = PredictionHead(
            input_dim=d_model,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: [B, T, F] -> pred_score [B].

        Args:
            x: Input feature sequence of shape [B, T, F].

        Returns:
            Prediction score tensor of shape [B].
        """
        self._validate_input(x)

        z = self.input_proj(x)                    # [B, T, d_model]
        z = self.pos_encoder(z)                    # [B, T, d_model]

        if self.use_cls_token:
            cls = self.cls_token.expand(z.size(0), -1, -1)  # [B, 1, d_model]
            z = torch.cat([cls, z], dim=1)         # [B, T+1, d_model]

        encoded = self.encoder(z)                   # [B, T(+1), d_model]

        context = encoded[:, 0] if self.use_cls_token else encoded[:, -1]  # [B, d_model]
        context = self.context_norm(context)        # [B, d_model]
        return self.head(context)                    # [B]

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"TransformerStockModel expects a torch.Tensor, got {type(x).__name__}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"TransformerStockModel expects a floating point tensor, got {x.dtype}"
            )
        if x.ndim != 3:
            raise ValueError(
                f"TransformerStockModel expects [B, T, F], got shape {tuple(x.shape)}"
            )
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected feature dimension {self.num_features}, got {x.size(-1)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("TransformerStockModel input contains NaN or Inf values.")
