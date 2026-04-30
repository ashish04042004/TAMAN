import json

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from config import Config
from dataset import AirQualityDataset, StandardizationStats, get_image_transform
from model import MultiModalRegressor


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(cfg.model_output_path, map_location=device)
    stats = StandardizationStats(
        mean=torch.tensor(ckpt["metadata_mean"], dtype=torch.float32),
        std=torch.tensor(ckpt["metadata_std"], dtype=torch.float32),
    )

    test_ds = AirQualityDataset(
        csv_path=cfg.test_csv,
        image_root=cfg.image_root,
        target_col=cfg.target_col,
        transform=get_image_transform(train=False),
        stats=stats,
        metadata_columns=cfg.metadata_columns,
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MultiModalRegressor(metadata_dim=len(cfg.metadata_columns)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for images, metadata, y in test_loader:
            images = images.to(device)
            metadata = metadata.to(device)
            y_hat = model(images, metadata)
            preds.extend(y_hat.cpu().numpy().tolist())
            targets.extend(y.numpy().tolist())

    metrics = {
        "mae": mean_absolute_error(targets, preds),
        "rmse": mean_squared_error(targets, preds) ** 0.5,
        "r2": r2_score(targets, preds),
    }
    with open(cfg.metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            f"target_{cfg.target_col}": targets,
            f"pred_{cfg.target_col}": preds,
        }
    )
    pred_df.to_csv(cfg.predictions_output_path, index=False)
    print(metrics)


if __name__ == "__main__":
    main()
