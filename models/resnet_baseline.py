from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        # Gracefully fall back to random init if pretrained weights are unavailable (e.g., offline)
        weights = None
        if pretrained:
            try:
                weights = ResNet18_Weights.IMAGENET1K_V1
            except Exception:
                weights = None
        try:
            self.backbone = resnet18(weights=weights)
        except Exception:
            self.backbone = resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if not name.startswith('fc.'):
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
