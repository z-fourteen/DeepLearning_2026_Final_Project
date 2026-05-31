from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.training.metrics import summarize_daily_ic


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
    return resolved


class Trainer:
    """Minimal trainer for sequence stock prediction models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        scheduler: Any | None = None,
        config: Mapping[str, Any] | None = None,
        device: str | torch.device = "auto",
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = dict(config or {})
        self.max_grad_norm = float(self.config.get("max_grad_norm", 1.0))
        self.early_stop_metric = str(self.config.get("early_stop_metric", "rank_ic_mean"))
        self.early_stop_patience = int(self.config.get("early_stop_patience", 10))
        self.collapse_stop_patience = int(self.config.get("collapse_stop_patience", 2))
        collapse_statuses = self.config.get("collapse_stop_statuses", ["prediction_collapse"])
        if isinstance(collapse_statuses, str):
            collapse_statuses = [collapse_statuses]
        self.collapse_stop_statuses = {str(status) for status in collapse_statuses}
        self.min_daily_count = int(self.config.get("min_daily_count", 20))
        self.min_best_daily_ratio = float(self.config.get("min_best_daily_ratio", 0.8))
        self.checkpoint_rank_ic_weight = float(self.config.get("checkpoint_rank_ic_weight", 1.0))
        self.checkpoint_topk_weight = float(self.config.get("checkpoint_topk_weight", 0.0))
        self.checkpoint_dispersion_weight = float(self.config.get("checkpoint_dispersion_weight", 0.0))
        self.checkpoint_topk = int(self.config.get("checkpoint_topk", 20))
        self.checkpoint_dispersion_floor = float(self.config.get("checkpoint_dispersion_floor", 0.0))
        self.history: list[dict[str, float | int | str]] = []
        self.best_metric = float("-inf")
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.best_epoch = -1
        self.stop_reason = "max_epochs_reached"

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True).view(-1)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x).view(-1)
            loss = self._compute_loss(pred, y, batch)
            loss.backward()
            if self.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            batch_size = int(y.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size

        return {"train_loss": total_loss / max(total_samples, 1)}

    @torch.no_grad()
    def validate(self) -> dict[str, float | int | str]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_pred: list[torch.Tensor] = []
        all_target: list[torch.Tensor] = []
        all_dates: list[str] = []

        for batch in self.val_loader:
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True).view(-1)
            pred = self.model(x).view(-1)
            loss = self._compute_loss(pred, y, batch)

            batch_size = int(y.numel())
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            all_pred.append(pred.detach().cpu())
            all_target.append(y.detach().cpu())
            all_dates.extend([str(date) for date in batch["trade_date"]])

        pred_tensor = torch.cat(all_pred) if all_pred else torch.empty(0)
        target_tensor = torch.cat(all_target) if all_target else torch.empty(0)
        metrics = summarize_daily_ic(
            pred_tensor,
            target_tensor,
            all_dates,
            min_count=self.min_daily_count,
        )
        metrics["val_loss"] = total_loss / max(total_samples, 1)
        metrics.update(self._prediction_diagnostics(pred_tensor, target_tensor))
        metrics.update(self._checkpoint_diagnostics(pred_tensor, target_tensor, all_dates, metrics))
        return metrics

    def fit(self, max_epochs: int, checkpoint_path: str | Path | None = None) -> list[dict[str, float | int | str]]:
        stale_epochs = 0
        collapse_epochs = 0

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            if self.scheduler is not None:
                self.scheduler.step()

            record: dict[str, float | int | str] = {"epoch": epoch, **train_metrics, **val_metrics}
            self.history.append(record)

            current = float(record.get(self.early_stop_metric, float("nan")))
            daily_coverage_ratio = self._daily_coverage_ratio(record)
            checkpoint_eligible = (
                math.isfinite(current)
                and daily_coverage_ratio >= self.min_best_daily_ratio
            )
            record["daily_coverage_ratio"] = daily_coverage_ratio
            record["checkpoint_eligible"] = int(checkpoint_eligible)

            if checkpoint_eligible and current > self.best_metric:
                self.best_metric = current
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
                if checkpoint_path is not None:
                    path = Path(checkpoint_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.best_state_dict,
                            "metric": self.best_metric,
                            "history": self.history,
                        },
                        path,
                    )
            else:
                stale_epochs += 1

            daily_status = str(record.get("daily_status", ""))
            if daily_status in self.collapse_stop_statuses:
                collapse_epochs += 1
            else:
                collapse_epochs = 0

            record["stale_epochs"] = stale_epochs
            record["collapse_epochs"] = collapse_epochs

            if self.collapse_stop_patience > 0 and collapse_epochs >= self.collapse_stop_patience:
                self.stop_reason = f"collapse_early_stop:{daily_status}"
                break

            if stale_epochs >= self.early_stop_patience:
                self.stop_reason = f"metric_early_stop:{self.early_stop_metric}"
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        return self.history

    def _daily_coverage_ratio(self, record: Mapping[str, Any]) -> float:
        daily_count = int(record.get("daily_count", 0) or 0)
        eligible_daily_count = int(record.get("eligible_daily_count", 0) or 0)
        if eligible_daily_count <= 0:
            return 0.0
        return float(daily_count / eligible_daily_count)

    def _compute_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        batch: Mapping[str, Any],
    ) -> torch.Tensor:
        try:
            return self.loss_fn(pred, target, batch.get("trade_date"))
        except TypeError:
            return self.loss_fn(pred, target)

    def _prediction_diagnostics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, float | int]:
        if pred.numel() == 0:
            return {
                "pred_count": 0,
                "valid_pred_ratio": float("nan"),
                "pred_mean": float("nan"),
                "pred_std": float("nan"),
                "pred_min": float("nan"),
                "pred_max": float("nan"),
                "target_mean": float("nan"),
                "target_std": float("nan"),
            }

        pred = pred.float()
        target = target.float()
        finite_pred = torch.isfinite(pred)
        finite_target = torch.isfinite(target)
        finite_both = finite_pred & finite_target
        valid_pred = pred[finite_pred]
        valid_target = target[finite_target]

        def stat(values: torch.Tensor, op: str) -> float:
            if values.numel() == 0:
                return float("nan")
            if op == "std" and values.numel() < 2:
                return float("nan")
            return float(getattr(values, op)().item())

        return {
            "pred_count": int(pred.numel()),
            "valid_pred_ratio": float(finite_both.float().mean().item()),
            "pred_mean": stat(valid_pred, "mean"),
            "pred_std": stat(valid_pred, "std"),
            "pred_min": stat(valid_pred, "min"),
            "pred_max": stat(valid_pred, "max"),
            "target_mean": stat(valid_target, "mean"),
            "target_std": stat(valid_target, "std"),
        }

    def _checkpoint_diagnostics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        trade_dates: list[str],
        metrics: Mapping[str, Any],
    ) -> dict[str, float]:
        rank_ic = float(metrics.get("rank_ic_mean", float("nan")))
        rank_ic_for_score = rank_ic if math.isfinite(rank_ic) else 0.0
        if pred.numel() == 0 or len(trade_dates) != pred.numel():
            score = self.checkpoint_rank_ic_weight * rank_ic_for_score
            return {
                "topk_proxy_mean": float("nan"),
                "dispersion_floor_ratio": float("nan"),
                "checkpoint_score": float(score),
            }

        pred = pred.float()
        target = target.float()
        finite = torch.isfinite(pred) & torch.isfinite(target)
        topk_returns: list[torch.Tensor] = []
        dispersion_pass: list[float] = []
        date_keys = [str(date) for date in trade_dates]
        for date in sorted(set(date_keys)):
            mask = pred.new_tensor([key == date for key in date_keys], dtype=torch.bool) & finite
            day_pred = pred[mask]
            day_target = target[mask]
            if day_pred.numel() < max(2, min(self.checkpoint_topk, self.min_daily_count)):
                continue
            k = min(self.checkpoint_topk, day_pred.numel())
            top_idx = torch.topk(day_pred, k=k, largest=True).indices
            topk_returns.append(day_target[top_idx].mean())
            if day_pred.numel() >= 2:
                dispersion = float(day_pred.std(unbiased=True).item())
                dispersion_pass.append(float(dispersion >= self.checkpoint_dispersion_floor))

        topk_proxy = torch.stack(topk_returns).mean().item() if topk_returns else float("nan")
        dispersion_ratio = float(sum(dispersion_pass) / len(dispersion_pass)) if dispersion_pass else float("nan")
        topk_for_score = topk_proxy if math.isfinite(topk_proxy) else 0.0
        dispersion_for_score = dispersion_ratio if math.isfinite(dispersion_ratio) else 0.0
        score = (
            self.checkpoint_rank_ic_weight * rank_ic_for_score
            + self.checkpoint_topk_weight * topk_for_score
            + self.checkpoint_dispersion_weight * dispersion_for_score
        )
        return {
            "topk_proxy_mean": float(topk_proxy),
            "dispersion_floor_ratio": float(dispersion_ratio),
            "checkpoint_score": float(score),
        }
