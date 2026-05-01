from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
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


def parse_frame_id_from_image_name(image_name: str) -> int:
    """Integer frame/sample id from paths like traqid_subset/front/1399.png."""
    base = os.path.basename(str(image_name))
    stem, _ = os.path.splitext(base)
    if stem.isdigit():
        return int(stem)
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else 0


class TemporalAirQualityDataset(Dataset):
    """
    Sliding windows over rows sorted by frame id (proxy for capture order on TRAQID-style names).

    Returns a sequence of `seq_len` images, metadata (z-scored) from the **last** timestep,
    target AQI at the last timestep, and raw (is_night, season_code) from the last row for analysis exports.
    """

    def __init__(
        self,
        csv_path: str,
        image_root: str,
        target_col: str,
        transform: transforms.Compose,
        stats: StandardizationStats,
        metadata_columns: Sequence[str],
        seq_len: int = 4,
        max_consecutive_gap: int = 250,
    ) -> None:
        self.df = pd.read_csv(csv_path).copy()
        self.image_root = image_root
        self.target_col = target_col
        self.transform = transform
        self.stats = stats
        self.metadata_columns = list(metadata_columns)
        self.seq_len = seq_len
        self.max_consecutive_gap = max_consecutive_gap

        self.df["_fid"] = self.df["image_name"].map(parse_frame_id_from_image_name)
        self.df = self.df.sort_values("_fid", kind="mergesort").reset_index(drop=True)
        ids = self.df["_fid"].to_numpy(dtype=np.int64)
        self._starts: list[int] = []
        n = len(self.df)
        for i in range(0, n - seq_len + 1):
            ok = True
            for k in range(1, seq_len):
                if int(ids[i + k]) - int(ids[i + k - 1]) > max_consecutive_gap:
                    ok = False
                    break
            if ok:
                self._starts.append(i)

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int):
        start = self._starts[idx]
        imgs = []
        for k in range(self.seq_len):
            row = self.df.iloc[start + k]
            img_path = os.path.join(self.image_root, row["image_name"])
            image = Image.open(img_path).convert("RGB")
            imgs.append(self.transform(image))
        image_seq = torch.stack(imgs, dim=0)  # (T, C, H, W)

        last = self.df.iloc[start + self.seq_len - 1]
        meta_values = (
            pd.to_numeric(last[self.metadata_columns], errors="coerce").fillna(0.0).astype(float).values
        )
        meta = torch.tensor(meta_values, dtype=torch.float32)
        meta = (meta - self.stats.mean) / self.stats.std

        target = torch.tensor(float(last[self.target_col]), dtype=torch.float32)

        aux = torch.tensor(
            [float(last.get("is_night", 0.0)), float(last.get("season_code", 0.0))],
            dtype=torch.float32,
        )
        return image_seq, meta, target, aux
