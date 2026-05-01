import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import (
    AirQualityDataset,
    TemporalAirQualityDataset,
    compute_stats,
    get_image_transform,
)
from metrics_eval import regression_metrics_with_cpcb_f1
from model import MultiModalRegressor
from model_taman import TAMAN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_taman: bool):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            if use_taman:
                images, metadata, y, _aux = batch
            else:
                images, metadata, y = batch
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

    return regression_metrics_with_cpcb_f1(targets, preds)


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
    if device.type != "cuda":
        print("WARNING: CUDA not available; training on CPU will be very slow for TAMAN/CNN.")

    import pandas as pd

    train_df = pd.read_csv(cfg.train_csv)
    stats = compute_stats(train_df, cfg.metadata_columns)

    use_taman = cfg.use_taman
    if use_taman:
        print(
            f"TAMAN: backbone={cfg.taman_backbone}, seq_len={cfg.taman_seq_len}, "
            f"device={device}, batch_size={cfg.batch_size}"
        )
    if use_taman:
        train_ds = TemporalAirQualityDataset(
            csv_path=cfg.train_csv,
            image_root=cfg.image_root,
            target_col=cfg.target_col,
            transform=get_image_transform(train=True),
            stats=stats,
            metadata_columns=cfg.metadata_columns,
            seq_len=cfg.taman_seq_len,
            max_consecutive_gap=cfg.taman_max_consecutive_frame_gap,
        )
        val_ds = TemporalAirQualityDataset(
            csv_path=cfg.val_csv,
            image_root=cfg.image_root,
            target_col=cfg.target_col,
            transform=get_image_transform(train=False),
            stats=stats,
            metadata_columns=cfg.metadata_columns,
            seq_len=cfg.taman_seq_len,
            max_consecutive_gap=cfg.taman_max_consecutive_frame_gap,
        )
        model = TAMAN(
            metadata_dim=len(cfg.metadata_columns),
            seq_len=cfg.taman_seq_len,
            backbone=cfg.taman_backbone,
            lstm_hidden=cfg.taman_lstm_hidden,
            lstm_layers=cfg.taman_lstm_layers,
            meta_embed_dim=cfg.taman_meta_embed_dim,
            fusion_dim=cfg.taman_fusion_dim,
            head_hidden=cfg.taman_head_hidden,
        ).to(device)
        ckpt_extra = {
            "model_type": "taman",
            "seq_len": cfg.taman_seq_len,
            "backbone": cfg.taman_backbone,
            "max_consecutive_gap": cfg.taman_max_consecutive_frame_gap,
        }
        model_out = cfg.taman_model_output_path
        history_out = cfg.taman_train_history_output_path
        print(f"TAMAN windows: train={len(train_ds)}, val={len(val_ds)}")
    else:
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
        model = MultiModalRegressor(
            metadata_dim=len(cfg.metadata_columns),
            backbone=cfg.multimodal_backbone,
        ).to(device)
        ckpt_extra = {"model_type": "multimodal", "backbone": cfg.multimodal_backbone}
        model_out = cfg.model_output_path
        history_out = cfg.train_history_output_path

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

    if use_taman and (len(train_ds) == 0 or len(val_ds) == 0):
        raise RuntimeError(
            "Temporal dataset has zero valid windows. "
            "Try increasing `taman_max_consecutive_frame_gap` in config, "
            "or disable `use_taman` if your CSV frame ids are not ordered numerically."
        )

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
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            if use_taman:
                images, metadata, y, _aux = batch
            else:
                images, metadata, y = batch
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

        val_metrics = evaluate(model, val_loader, device, use_taman)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                **val_metrics,
            }
        )
        tqdm.write(
            f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
            f"val_mae={val_metrics['mae']:.4f}, val_rmse={val_metrics['rmse']:.4f}, "
            f"val_r2={val_metrics['r2']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )
        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            payload = {
                "model_state_dict": model.state_dict(),
                "metadata_mean": stats.mean.tolist(),
                "metadata_std": stats.std.tolist(),
                **ckpt_extra,
            }
            torch.save(payload, model_out)

    with open(history_out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
