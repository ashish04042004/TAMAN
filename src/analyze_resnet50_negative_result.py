"""
Reproducible analysis for the V3 (ResNet18) vs V4 (ResNet50) negative result.

Run from project root:
    python src/analyze_resnet50_negative_result.py

Writes: outputs/analysis_resnet50_negative_result.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

# Ensure imports resolve like other src/ scripts when run as python src/...py from repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import AirQualityDataset, StandardizationStats, get_image_transform
from model import MultiModalRegressor


@dataclass
class MultimodalParamCount:
    backbone: str
    total_trainable_params: int
    image_branch_params: int
    head_params: int


def _meta_encoder(metadata_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(metadata_dim, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Dropout(0.2),
        nn.Linear(64, 64),
        nn.ReLU(),
    )


def _fusion_and_regressor(image_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> tuple[nn.Sequential, nn.Sequential]:
    fusion_gate = nn.Sequential(
        nn.Linear(image_dim + 64, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Softmax(dim=1),
    )
    regressor = nn.Sequential(
        nn.Linear(image_dim + 64, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return fusion_gate, regressor


def count_multimodal_params(backbone_name: str, metadata_dim: int = 8) -> MultimodalParamCount:
    if backbone_name == "resnet18":
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif backbone_name == "resnet50":
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(backbone_name)
    image_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    meta = _meta_encoder(metadata_dim)
    fusion, reg = _fusion_and_regressor(image_dim)
    parts = nn.ModuleList([backbone, meta, fusion, reg])
    total = sum(p.numel() for p in parts.parameters() if p.requires_grad)
    img_only = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    head = total - img_only
    return MultimodalParamCount(
        backbone=backbone_name,
        total_trainable_params=total,
        image_branch_params=img_only,
        head_params=head,
    )


def _history_sanity(history: list[dict]) -> dict:
    rmses = [float(h["rmse"]) for h in history]
    unique = len(set(round(x, 6) for x in rmses))
    return {
        "epochs": len(history),
        "unique_val_rmse_rounded_6dp": unique,
        "looks_degenerate": unique <= 2 and len(history) >= 5,
    }


def _v4_overfitting_note(history: list[dict]) -> dict:
    """Last-epoch val RMSE worsening while train loss drops is a classic overfitting signal."""
    last = history[-1]
    prev = history[-2]
    return {
        "epoch_last": int(last["epoch"]),
        "train_loss_last": float(last["train_loss"]),
        "val_rmse_last": float(last["rmse"]),
        "epoch_prev": int(prev["epoch"]),
        "train_loss_prev": float(prev["train_loss"]),
        "val_rmse_prev": float(prev["rmse"]),
        "train_loss_decreased_last_epoch": float(last["train_loss"]) < float(prev["train_loss"]),
        "val_rmse_increased_last_epoch": float(last["rmse"]) > float(prev["rmse"]),
    }


@torch.no_grad()
def fusion_gate_summary(cfg: Config, ckpt_path: str, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    stats = StandardizationStats(
        mean=torch.tensor(ckpt["metadata_mean"], dtype=torch.float32),
        std=torch.tensor(ckpt["metadata_std"], dtype=torch.float32),
    )
    val_ds = AirQualityDataset(
        csv_path=cfg.val_csv,
        image_root=cfg.image_root,
        target_col=cfg.target_col,
        transform=get_image_transform(train=False),
        stats=stats,
        metadata_columns=cfg.metadata_columns,
    )
    loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = MultiModalRegressor(metadata_dim=len(cfg.metadata_columns)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_w: list[torch.Tensor] = []
    meta_w: list[torch.Tensor] = []
    for images, metadata, _ in loader:
        images = images.to(device)
        metadata = metadata.to(device)
        img_feat = model.image_encoder(images)
        meta_feat = model.meta_encoder(metadata)
        combined = torch.cat([img_feat, meta_feat], dim=1)
        gates = model.fusion_gate(combined)
        img_w.append(gates[:, 0].detach().cpu())
        meta_w.append(gates[:, 1].detach().cpu())

    iw = torch.cat(img_w)
    mw = torch.cat(meta_w)
    im = float(iw.mean())
    mm = float(mw.mean())
    if im >= 0.65:
        hint = "Gate skewed toward the image branch: metadata is down-weighted for most validation samples (check for fusion saturation / optimization dynamics)."
    elif mm >= 0.65:
        hint = "Gate skewed toward the metadata branch: images contribute less in the gated representation for most validation samples."
    else:
        hint = "Gate weights are relatively balanced between modalities on average."
    return {
        "split": "validation",
        "n_batches": len(loader),
        "image_gate_mean": im,
        "image_gate_std": float(iw.std()),
        "metadata_gate_mean": mm,
        "metadata_gate_std": float(mw.std()),
        "interpretation_hint": hint,
    }


def main() -> None:
    cfg = Config()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.join(root, "outputs", "analysis_resnet50_negative_result.json")

    metrics_v3 = os.path.join(root, "outputs", "eval", "v3", "test_metrics.json")
    metrics_v4 = os.path.join(root, "outputs", "eval", "v4", "test_metrics.json")
    hist_v3 = os.path.join(root, "outputs", "train_history_v3.json")
    hist_v4 = os.path.join(root, "outputs", "train_history_v4_resnet50.json")
    ckpt_v4 = os.path.join(root, cfg.model_output_path.replace("/", os.sep))

    with open(metrics_v3, "r", encoding="utf-8") as f:
        m3 = json.load(f)
    with open(metrics_v4, "r", encoding="utf-8") as f:
        m4 = json.load(f)

    with open(hist_v3, "r", encoding="utf-8") as f:
        h3 = json.load(f)
    with open(hist_v4, "r", encoding="utf-8") as f:
        h4 = json.load(f)

    meta_dim = len(cfg.metadata_columns)
    pc18 = count_multimodal_params("resnet18", meta_dim)
    pc50 = count_multimodal_params("resnet50", meta_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gates_block: dict | None = None
    if os.path.isfile(ckpt_v4):
        gates_block = fusion_gate_summary(cfg, ckpt_v4, device)
    else:
        gates_block = {"error": f"Checkpoint not found: {ckpt_v4}"}

    report = {
        "title": "ResNet50 (V4) vs ResNet18 (V3) negative result — quantitative hooks",
        "test_metrics": {"v3_resnet18": m3, "v4_resnet50": m4},
        "test_metric_deltas_v4_minus_v3": {
            "mae": float(m4["mae"] - m3["mae"]),
            "rmse": float(m4["rmse"] - m3["rmse"]),
            "r2": float(m4["r2"] - m3["r2"]),
        },
        "parameter_counts": {
            "v3_style_resnet18": asdict(pc18),
            "v4_style_resnet50": asdict(pc50),
            "total_params_ratio_resnet50_over_resnet18": float(pc50.total_trainable_params)
            / float(pc18.total_trainable_params),
        },
        "train_history_sanity": {
            "v3": _history_sanity(h3),
            "v4": _history_sanity(h4),
        },
        "v4_last_epoch_overfitting_signal": _v4_overfitting_note(h4),
        "v4_validation_curve_tail": [h4[-3], h4[-2], h4[-1]],
        "fusion_gate_on_validation_best_checkpoint": gates_block,
        "caveats": [
            "Saved train_history_v3.json may be degenerate (constant val metrics across epochs). Do not use it for curve-based claims until re-exported from a clean run.",
            "V4 checkpoint on disk is the best-validation-RMSE save from training; test metrics were produced with evaluate.py loading that checkpoint.",
        ],
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
