"""ResNet-18 regression model for relative upper-body depth."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18RelativeDepthEstimator(nn.Module):
    """Estimate one waist-relative depth value per configured keypoint."""

    def __init__(self, num_keypoints: int, pretrained: bool = True) -> None:
        super().__init__()
        if num_keypoints < 1:
            raise ValueError("num_keypoints must be positive.")
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_keypoints),
        )

    def forward(self, image: Tensor) -> Tensor:
        """Return a ``(batch, num_keypoints)`` depth tensor."""
        return self.head(self.pool(self.features(image)))

