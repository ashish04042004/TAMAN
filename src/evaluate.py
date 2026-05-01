import argparse
import json
import os
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import Config
from dataset import AirQualityDataset, TemporalAirQualityDataset, StandardizationStats, get_image_transform
from metrics_eval import regression_metrics_with_cpcb_f1
from model import MultiModalRegressor
from model_taman import TAMAN


def evaluate_checkpoint(
    cfg: Config,
    *,
    ckpt_path: str,
    metrics_path: str,
    pred_path: str,
    multimodal_backbone: str = "resnet50",
) -> dict[str, Any]:
    """
    Run test-set evaluation and write metrics + predictions CSV.
    For non-TAMAN checkpoints, ``multimodal_backbone`` must match the trained architecture (resnet18 vs resnet50).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    stats = StandardizationStats(
        mean=torch.tensor(ckpt["metadata_mean"], dtype=torch.float32),
        std=torch.tensor(ckpt["metadata_std"], dtype=torch.float32),
    )

    model_type = ckpt.get("model_type", "multimodal_resnet50")
    use_taman = model_type == "taman"

    if use_taman:
        seq_len = int(ckpt.get("seq_len", cfg.taman_seq_len))
        backbone = ckpt.get("backbone", cfg.taman_backbone)
        max_gap = int(ckpt.get("max_consecutive_gap", cfg.taman_max_consecutive_frame_gap))
        test_ds = TemporalAirQualityDataset(
            csv_path=cfg.test_csv,
            image_root=cfg.image_root,
            target_col=cfg.target_col,
            transform=get_image_transform(train=False),
            stats=stats,
            metadata_columns=cfg.metadata_columns,
            seq_len=seq_len,
            max_consecutive_gap=max_gap,
        )
        model = TAMAN(
            metadata_dim=len(cfg.metadata_columns),
            seq_len=seq_len,
            backbone=str(backbone),
            lstm_hidden=cfg.taman_lstm_hidden,
            lstm_layers=cfg.taman_lstm_layers,
            meta_embed_dim=cfg.taman_meta_embed_dim,
            fusion_dim=cfg.taman_fusion_dim,
            head_hidden=cfg.taman_head_hidden,
        ).to(device)
    else:
        bb = str(ckpt.get("backbone", multimodal_backbone)).lower()
        ck_meta_dim = int(ckpt["model_state_dict"]["meta_encoder.0.weight"].shape[1])
        cfg_meta_dim = len(cfg.metadata_columns)
        if ck_meta_dim != cfg_meta_dim:
            raise ValueError(
                f"Checkpoint expects metadata_dim={ck_meta_dim} but config has {cfg_meta_dim} columns "
                f"{cfg.metadata_columns!r}. Retrain the checkpoint with the current schema or adjust config."
            )
        test_ds = AirQualityDataset(
            csv_path=cfg.test_csv,
            image_root=cfg.image_root,
            target_col=cfg.target_col,
            transform=get_image_transform(train=False),
            stats=stats,
            metadata_columns=cfg.metadata_columns,
        )
        model = MultiModalRegressor(
            metadata_dim=cfg_meta_dim,
            backbone=bb,
        ).to(device)

    if len(test_ds) == 0:
        raise RuntimeError(
            "Test set produced zero samples (check temporal windows / frame gaps). "
            "Increase `taman_max_consecutive_frame_gap` or use non-temporal evaluate."
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    preds, targets = [], []
    nights, seasons = [], []
    with torch.no_grad():
        for batch in test_loader:
            if use_taman:
                images, metadata, y, aux = batch
                nights.extend(aux[:, 0].cpu().numpy().tolist())
                seasons.extend(aux[:, 1].cpu().numpy().tolist())
            else:
                images, metadata, y = batch
            images = images.to(device)
            metadata = metadata.to(device)
            y_hat = model(images, metadata)
            preds.extend(y_hat.cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())

    metrics = regression_metrics_with_cpcb_f1(targets, preds)
    metrics["checkpoint"] = os.path.normpath(ckpt_path).replace("\\", "/")
    metrics["model_type"] = model_type

    os.makedirs(os.path.dirname(metrics_path) or ".", exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rows = {
        f"target_{cfg.target_col}": targets,
        f"pred_{cfg.target_col}": preds,
    }
    if use_taman and len(nights) == len(targets):
        rows["is_night"] = nights
        rows["season_code"] = seasons
    pred_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    pred_df.to_csv(pred_path, index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multimodal or TAMAN checkpoint on the test split.")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint (defaults from config)")
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Where to write test_metrics.json (default: config metrics path)",
    )
    parser.add_argument(
        "--pred-out",
        default=None,
        help="Where to write predictions CSV (default: config predictions path)",
    )
    parser.add_argument(
        "--backbone",
        choices=("resnet18", "resnet50"),
        default=None,
        help="For non-TAMAN checkpoints missing 'backbone' in file: architecture to instantiate (default: resnet50)",
    )
    args = parser.parse_args()

    cfg = Config()
    ckpt_path = args.checkpoint or (cfg.taman_model_output_path if cfg.use_taman else cfg.model_output_path)
    metrics_path = args.metrics_out or (
        cfg.taman_metrics_output_path if cfg.use_taman else cfg.metrics_output_path
    )
    pred_path = args.pred_out or (cfg.taman_predictions_output_path if cfg.use_taman else cfg.predictions_output_path)
    multimodal_bb = args.backbone or cfg.multimodal_backbone

    metrics = evaluate_checkpoint(
        cfg,
        ckpt_path=ckpt_path,
        metrics_path=metrics_path,
        pred_path=pred_path,
        multimodal_backbone=multimodal_bb,
    )
    print(metrics)


if __name__ == "__main__":
    main()
