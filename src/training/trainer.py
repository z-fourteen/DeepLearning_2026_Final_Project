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
            loss = self.loss_fn(pred, y)
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
            loss = self.loss_fn(pred, y)

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
            if math.isfinite(current) and current > self.best_metric:
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
