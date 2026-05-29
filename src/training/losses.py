from __future__ import annotations

import torch
from torch import nn


def _as_date_key(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class PearsonICLoss(nn.Module):
    """Negative Pearson correlation loss for 1D prediction targets."""

    def __init__(self, eps: float = 1e-8, min_samples: int = 2):
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        self.eps = float(eps)
        self.min_samples = int(min_samples)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        finite_mask = torch.isfinite(pred) & torch.isfinite(target)
        pred = pred[finite_mask]
        target = target[finite_mask]

        if pred.numel() < self.min_samples:
            return pred.new_tensor(0.0)

        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        covariance = torch.sum(pred_centered * target_centered)
        pred_scale = torch.sqrt(torch.sum(pred_centered.square()) + self.eps)
        target_scale = torch.sqrt(torch.sum(target_centered.square()) + self.eps)
        corr = covariance / (pred_scale * target_scale)
        return -corr.clamp(min=-1.0, max=1.0)


class MSEICLoss(nn.Module):
    """Linear combination of MSE and differentiable Pearson IC objective.

    Loss = (1 - alpha) * MSE - alpha * PearsonCorr(pred, target)
    """

    def __init__(self, alpha: float = 0.1, eps: float = 1e-8, min_samples: int = 2):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.mse = nn.MSELoss()
        self.ic = PearsonICLoss(eps=eps, min_samples=min_samples)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trade_date: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        mse_loss = self.mse(pred, target)
        ic_loss = self._daily_ic_loss(pred, target, trade_date) if trade_date is not None else self.ic(pred, target)
        return (1.0 - self.alpha) * mse_loss + self.alpha * ic_loss

    def _daily_ic_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trade_date: list[str] | tuple[str, ...],
    ) -> torch.Tensor:
        if len(trade_date) != pred.numel():
            raise ValueError(
                f"trade_date length ({len(trade_date)}) must match prediction length ({pred.numel()})"
            )

        losses: list[torch.Tensor] = []
        date_keys = [_as_date_key(item) for item in trade_date]
        for date in sorted(set(date_keys)):
            mask = pred.new_tensor([key == date for key in date_keys], dtype=torch.bool)
            if int(mask.sum().item()) >= self.ic.min_samples:
                losses.append(self.ic(pred[mask], target[mask]))

        if not losses:
            return pred.new_tensor(0.0)
        return torch.stack(losses).mean()
