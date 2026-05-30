from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class TopKMarginICLoss(nn.Module):
    """Daily top-k ranking loss aligned with long-only portfolio formation.

    The objective rewards high IC, but pushes the true forward-return winners
    above the model's high-scored false positives by a configurable margin.
    """

    def __init__(
        self,
        k: int = 20,
        negative_multiplier: int = 3,
        margin: float = 0.02,
        temperature: float = 0.01,
        ic_alpha: float = 0.2,
        mse_alpha: float = 0.02,
        eps: float = 1e-8,
        min_samples: int = 20,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if negative_multiplier <= 0:
            raise ValueError(f"negative_multiplier must be positive, got {negative_multiplier}")
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if ic_alpha < 0 or mse_alpha < 0:
            raise ValueError("ic_alpha and mse_alpha must be non-negative")
        self.k = int(k)
        self.negative_multiplier = int(negative_multiplier)
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.ic_alpha = float(ic_alpha)
        self.mse_alpha = float(mse_alpha)
        self.ic = PearsonICLoss(eps=eps, min_samples=min_samples)
        self.mse = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trade_date: list[str] | tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        if trade_date is None:
            ranking_loss = self._single_cross_section_loss(pred, target)
            return self._combine(pred, target, ranking_loss)

        if len(trade_date) != pred.numel():
            raise ValueError(
                f"trade_date length ({len(trade_date)}) must match prediction length ({pred.numel()})"
            )

        date_keys = [_as_date_key(item) for item in trade_date]
        losses: list[torch.Tensor] = []
        for date in sorted(set(date_keys)):
            mask = pred.new_tensor([key == date for key in date_keys], dtype=torch.bool)
            if int(mask.sum().item()) >= max(2, self.k + 1):
                losses.append(self._single_cross_section_loss(pred[mask], target[mask]))

        if not losses:
            ranking_loss = pred.new_tensor(0.0)
        else:
            ranking_loss = torch.stack(losses).mean()
        return self._combine(pred, target, ranking_loss)

    def _single_cross_section_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(pred) & torch.isfinite(target)
        pred = pred[finite_mask]
        target = target[finite_mask]
        n = pred.numel()
        if n < 2:
            return pred.new_tensor(0.0)

        k_pos = min(self.k, max(1, n // 2))
        remaining = n - k_pos
        k_neg = min(max(self.k * self.negative_multiplier, k_pos), remaining)
        if k_neg <= 0:
            return pred.new_tensor(0.0)

        true_top_idx = torch.topk(target, k=k_pos, largest=True).indices
        pred_hard_idx = torch.topk(pred, k=k_neg + k_pos, largest=True).indices
        true_mask = torch.zeros(n, dtype=torch.bool, device=pred.device)
        true_mask[true_top_idx] = True
        hard_negative_idx = pred_hard_idx[~true_mask[pred_hard_idx]][:k_neg]
        if hard_negative_idx.numel() == 0:
            target_bottom_idx = torch.topk(target, k=k_neg, largest=False).indices
            hard_negative_idx = target_bottom_idx

        pos_scores = pred[true_top_idx].view(-1, 1)
        neg_scores = pred[hard_negative_idx].view(1, -1)
        pairwise_gap = pos_scores - neg_scores
        margin_loss = F.softplus((self.margin - pairwise_gap) / self.temperature).mean() * self.temperature

        top_softmax = torch.softmax(pred / self.temperature, dim=0)
        portfolio_return = torch.sum(top_softmax * target)
        oracle_return = target[true_top_idx].mean()
        return margin_loss + F.relu(oracle_return - portfolio_return)

    def _combine(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ranking_loss: torch.Tensor,
    ) -> torch.Tensor:
        finite_mask = torch.isfinite(pred) & torch.isfinite(target)
        pred_clean = pred[finite_mask]
        target_clean = target[finite_mask]
        if pred_clean.numel() < 2:
            return ranking_loss

        ic_loss = self.ic(pred_clean, target_clean)
        mse_loss = self.mse(pred_clean, target_clean)
        return ranking_loss + self.ic_alpha * ic_loss + self.mse_alpha * mse_loss
