"""
Temporal Adaptive Multimodal AQI Network (TAMAN).

Image sequence -> CNN per frame -> LSTM -> temporal embedding;
metadata -> MLP -> weather embedding;
scalar sigmoid gate fuses the two branches -> AQI regression head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def _build_resnet_backbone(name: str) -> tuple[nn.Module, int]:
    name = name.lower()
    if name == "resnet18":
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif name == "resnet50":
        m = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown backbone: {name}. Use resnet18 or resnet50.")
    dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, dim


class TAMAN(nn.Module):
    def __init__(
        self,
        metadata_dim: int,
        seq_len: int = 4,
        backbone: str = "resnet18",
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        meta_embed_dim: int = 128,
        fusion_dim: int = 128,
        head_hidden: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.backbone_name = backbone.lower()
        self.image_encoder, image_dim = _build_resnet_backbone(self.backbone_name)

        self.lstm = nn.LSTM(
            input_size=image_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.temporal_proj = nn.Linear(lstm_hidden, fusion_dim)

        self.weather_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, meta_embed_dim),
            nn.ReLU(),
            nn.Linear(meta_embed_dim, fusion_dim),
        )

        self.gate_layer = nn.Linear(fusion_dim * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, image_sequence: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        # image_sequence: (B, T, C, H, W)
        b, t, c, h, w = image_sequence.shape
        if t != self.seq_len:
            raise ValueError(f"Expected sequence length {self.seq_len}, got {t}")

        x = image_sequence.reshape(b * t, c, h, w)
        feats = self.image_encoder(x)  # (B*T, D)
        feats = feats.view(b, t, -1)
        out, _ = self.lstm(feats)
        temporal_vec = out[:, -1, :]  # (B, lstm_hidden)
        temporal_emb = self.temporal_proj(temporal_vec)  # (B, fusion_dim)

        weather_emb = self.weather_mlp(metadata)  # (B, fusion_dim)

        fusion_in = torch.cat([temporal_emb, weather_emb], dim=1)
        gate = torch.sigmoid(self.gate_layer(fusion_in))  # (B, 1)
        fused = gate * temporal_emb + (1.0 - gate) * weather_emb

        pred = self.head(fused)
        return pred.squeeze(1)
