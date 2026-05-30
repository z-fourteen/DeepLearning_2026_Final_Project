from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DateBatchSampler, SequenceNPZDataset  # noqa: E402
from src.models import GRUStockModel, RegimeGatedGRUStockModel  # noqa: E402
from src.training import MSEICLoss, PearsonICLoss, TopKMarginICLoss, Trainer, resolve_device  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limited_loader(loader: DataLoader, max_batches: int | None):
    if max_batches is None:
        return loader
    return islice(loader, max_batches)


def build_loss(config: dict[str, Any]) -> torch.nn.Module:
    loss_name = str(config.get("loss_fn", "huber")).lower()
    if loss_name == "huber":
        return torch.nn.HuberLoss(delta=float(config.get("huber_delta", 1.0)))
    if loss_name == "mse":
        return torch.nn.MSELoss()
    if loss_name == "pearson_ic":
        return PearsonICLoss(
            eps=float(config.get("ic_loss_eps", 1e-8)),
            min_samples=int(config.get("ic_loss_min_samples", 2)),
        )
    if loss_name in {"mse_ic", "mse_pearson_ic"}:
        return MSEICLoss(
            alpha=float(config.get("ic_loss_alpha", 0.1)),
            eps=float(config.get("ic_loss_eps", 1e-8)),
            min_samples=int(config.get("ic_loss_min_samples", 2)),
        )
    if loss_name in {"topk_margin_ic", "topk_margin"}:
        return TopKMarginICLoss(
            k=int(config.get("topk_loss_k", 20)),
            negative_multiplier=int(config.get("topk_loss_negative_multiplier", 3)),
            margin=float(config.get("topk_loss_margin", 0.02)),
            temperature=float(config.get("topk_loss_temperature", 0.01)),
            ic_alpha=float(config.get("topk_loss_ic_alpha", 0.2)),
            mse_alpha=float(config.get("topk_loss_mse_alpha", 0.02)),
            eps=float(config.get("ic_loss_eps", 1e-8)),
            min_samples=int(config.get("ic_loss_min_samples", 20)),
        )
    raise ValueError(f"Unsupported loss_fn: {loss_name}")


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    max_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(config.get("scheduler", "none")).lower()
    if scheduler_name in {"none", "null"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_dataloader(
    dataset: SequenceNPZDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    batch_mode: str = "sample",
    seed: int = 42,
) -> DataLoader:
    if batch_mode == "date":
        sampler = DateBatchSampler(
            dataset,
            max_samples_per_batch=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    if batch_mode != "sample":
        raise ValueError(f"Unsupported batch_mode: {batch_mode}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


@torch.no_grad()
def predict_to_frame(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    model_name: str,
    max_batches: int | None = None,
) -> pd.DataFrame:
    model.eval()
    rows: list[pd.DataFrame] = []
    for batch in limited_loader(loader, max_batches):
        x = batch["x"].to(device, non_blocking=True)
        pred = model(x).detach().cpu().view(-1).numpy()
        label = batch["y"].detach().cpu().view(-1).numpy()
        rows.append(
            pd.DataFrame(
                {
                    "trade_date": [str(item) for item in batch["trade_date"]],
                    "ts_code": [str(item) for item in batch["ts_code"]],
                    "pred_score": pred,
                    "label_rel_return": label,
                    "split": split,
                    "model_name": model_name,
                }
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["trade_date", "ts_code", "pred_score", "label_rel_return", "split", "model_name"]
        )
    return pd.concat(rows, ignore_index=True)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sequence stock prediction models.")
    parser.add_argument("--config", required=True, help="Path to sequence training YAML config.")
    parser.add_argument("--device", help="Override config training.device: auto/cpu/cuda.")
    parser.add_argument("--output-dir", help="Override run.output_dir.")
    parser.add_argument("--dry-run", action="store_true", help="Build objects and print summary without training.")
    parser.add_argument("--max-epochs", type=int, help="Override max_epochs for smoke tests.")
    parser.add_argument("--max-train-batches", type=int, help="Limit train batches for smoke tests.")
    parser.add_argument("--max-val-batches", type=int, help="Limit validation batches for smoke tests.")
    parser.add_argument("--max-test-batches", type=int, help="Limit test prediction batches for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml(config_path)

    run_config = config.get("run", {})
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    seed = int(run_config.get("seed", 42))
    set_seed(seed)

    npz_path = Path(data_config["npz_path"])
    if not npz_path.is_absolute():
        npz_path = PROJECT_ROOT / npz_path

    train_dataset = SequenceNPZDataset(npz_path, str(data_config.get("train_split", "train")))
    val_dataset = SequenceNPZDataset(npz_path, str(data_config.get("validation_split", "validation")))
    test_dataset = SequenceNPZDataset(npz_path, str(data_config.get("test_split", "test")))

    batch_size = int(training_config.get("batch_size", 256))
    batch_mode = str(training_config.get("batch_mode", "sample")).lower()
    num_workers = int(data_config.get("num_workers", 0))
    device = resolve_device(args.device or training_config.get("device", "auto"))
    pin_memory = bool(data_config.get("pin_memory", True)) and device.type == "cuda"

    train_loader = build_dataloader(train_dataset, batch_size, True, num_workers, pin_memory, batch_mode, seed)
    val_loader = build_dataloader(val_dataset, batch_size, False, num_workers, pin_memory, batch_mode, seed)
    test_loader = build_dataloader(test_dataset, batch_size, False, num_workers, pin_memory, batch_mode, seed)

    model_name = str(model_config.get("name", "gru_baseline"))
    num_features = int(model_config.get("num_features", train_dataset.num_features))
    if num_features != train_dataset.num_features:
        raise ValueError(
            f"Config num_features={num_features} does not match dataset features={train_dataset.num_features}"
        )
    if model_name == "gru_baseline":
        model = GRUStockModel(num_features=num_features, config=model_config)
    elif model_name == "regime_gated_gru":
        model = RegimeGatedGRUStockModel(num_features=num_features, config=model_config)
    else:
        raise ValueError(f"Unsupported sequence model for this script version: {model_name}")
    optimizer = build_optimizer(model, training_config)
    loss_fn = build_loss(training_config)
    max_epochs = int(args.max_epochs or training_config.get("max_epochs", 80))
    scheduler = build_scheduler(optimizer, training_config, max_epochs)

    summary = {
        "run_name": run_config.get("name", "sequence_run"),
        "device": str(device),
        "npz_path": str(npz_path),
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "lookback": train_dataset.lookback,
        "num_features": train_dataset.num_features,
        "batch_size": batch_size,
        "batch_mode": batch_mode,
        "train_steps_per_epoch": len(train_loader),
        "validation_steps_per_epoch": len(val_loader),
        "test_steps": len(test_loader),
        "max_epochs": max_epochs,
        "model": model_name,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    output_dir = Path(args.output_dir or run_config.get("output_dir", f"outputs/runs/{summary['run_name']}"))
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(config_path, output_dir / "config.yaml")

    trainer = Trainer(
        model=model,
        train_loader=limited_loader(train_loader, args.max_train_batches),
        val_loader=limited_loader(val_loader, args.max_val_batches),
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        config=training_config,
        device=device,
    )
    history = trainer.fit(max_epochs=max_epochs, checkpoint_path=output_dir / "model.pt")

    if trainer.best_state_dict is not None:
        model.load_state_dict(trainer.best_state_dict)

    prediction_frames = [
        predict_to_frame(model, val_loader, device, "validation", model_name, args.max_val_batches),
        predict_to_frame(model, test_loader, device, "test", model_name, args.max_test_batches),
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)

    metrics = {
        "best_epoch": trainer.best_epoch,
        "best_metric": trainer.best_metric,
        "stop_reason": trainer.stop_reason,
        "history": history,
        "summary": summary,
        "prediction_rows": int(len(predictions)),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            json_safe(
                {
                    "output_dir": str(output_dir),
                    "best_epoch": trainer.best_epoch,
                    "best_metric": trainer.best_metric,
                }
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
