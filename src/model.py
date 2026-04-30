import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MultiModalRegressor(nn.Module):
    def __init__(self, metadata_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        image_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.meta_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(image_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(image_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_encoder(image)
        meta_feat = self.meta_encoder(metadata)
        combined = torch.cat([img_feat, meta_feat], dim=1)

        # Adaptive novelty: learn sample-wise modality importance.
        gates = self.fusion_gate(combined)
        img_weight = gates[:, 0:1]
        meta_weight = gates[:, 1:2]
        weighted = torch.cat([img_feat * img_weight, meta_feat * meta_weight], dim=1)

        pred = self.regressor(weighted)
        return pred.squeeze(1)
