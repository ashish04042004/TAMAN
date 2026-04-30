import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import (
    AirQualityDataset,
    compute_stats,
    get_image_transform,
)
from model import MultiModalRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for images, metadata, y in loader:
            images = images.to(device)
            metadata = metadata.to(device)
            y = y.to(device)
            if not torch.isfinite(images).all() or not torch.isfinite(metadata).all():
                raise ValueError("Non-finite values found in evaluation inputs.")
            y_hat = model(images, metadata)
            if not torch.isfinite(y_hat).all():
                raise ValueError("Model produced non-finite predictions during evaluation.")
            preds.extend(y_hat.cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds) ** 0.5
    r2 = r2_score(targets, preds)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def weighted_huber_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    delta: float,
    high_aqi_threshold: float,
    high_aqi_weight: float,
) -> torch.Tensor:
    err = torch.abs(y_pred - y_true)
    quadratic = 0.5 * (err**2)
    linear = delta * (err - 0.5 * delta)
    base = torch.where(err <= delta, quadratic, linear)
    weights = torch.where(
        y_true >= high_aqi_threshold,
        torch.full_like(y_true, high_aqi_weight),
        torch.ones_like(y_true),
    )
    return (base * weights).mean()


def main():
    cfg = Config()
    set_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd

    train_df = pd.read_csv(cfg.train_csv)
    stats = compute_stats(train_df, cfg.metadata_columns)

    train_ds = AirQualityDataset(
        csv_path=cfg.train_csv,
        image_root=cfg.image_root,
        target_col=cfg.target_col,
        transform=get_image_transform(train=True),
        stats=stats,
        metadata_columns=cfg.metadata_columns,
    )
    val_ds = AirQualityDataset(
        csv_path=cfg.val_csv,
        image_root=cfg.image_root,
        target_col=cfg.target_col,
        transform=get_image_transform(train=False),
        stats=stats,
        metadata_columns=cfg.metadata_columns,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = MultiModalRegressor(metadata_dim=len(cfg.metadata_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    use_amp = device.type == "cuda" and cfg.use_amp

    best_rmse = float("inf")
    history = []
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, metadata, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            images = images.to(device)
            metadata = metadata.to(device)
            y = y.to(device)
            if not torch.isfinite(images).all() or not torch.isfinite(metadata).all():
                raise ValueError("Non-finite values found in training inputs.")

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                y_hat = model(images, metadata)
                if not torch.isfinite(y_hat).all():
                    raise ValueError("Model produced non-finite predictions during training.")
                loss = weighted_huber_loss(
                    y_pred=y_hat,
                    y_true=y,
                    delta=cfg.huber_delta,
                    high_aqi_threshold=cfg.high_aqi_threshold,
                    high_aqi_weight=cfg.high_aqi_weight,
                )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss encountered during training.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **val_metrics,
            }
        )
        print(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}, val_rmse={val_metrics['rmse']:.4f}, "
            f"val_r2={val_metrics['r2']:.4f}"
        )
        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metadata_mean": stats.mean.tolist(),
                    "metadata_std": stats.std.tolist(),
                },
                cfg.model_output_path,
            )

    with open(cfg.train_history_output_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
