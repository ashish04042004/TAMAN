from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass
class StandardizationStats:
    mean: torch.Tensor
    std: torch.Tensor


def get_image_transform(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def compute_stats(df: pd.DataFrame, metadata_columns: Sequence[str]) -> StandardizationStats:
    meta = torch.tensor(df[list(metadata_columns)].values, dtype=torch.float32)
    mean = meta.mean(dim=0)
    std = meta.std(dim=0).clamp_min(1e-6)
    return StandardizationStats(mean=mean, std=std)


class AirQualityDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        target_col: str,
        transform: transforms.Compose,
        stats: StandardizationStats,
        metadata_columns: Sequence[str],
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.target_col = target_col
        self.transform = transform
        self.stats = stats
        self.metadata_columns = list(metadata_columns)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_name"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        meta_values = pd.to_numeric(row[self.metadata_columns], errors="coerce").fillna(0.0).astype(float).values
        meta = torch.tensor(meta_values, dtype=torch.float32)
        meta = (meta - self.stats.mean) / self.stats.std

        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return image, meta, target
