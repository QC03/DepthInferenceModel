"""Dataset and image augmentation utilities for relative upper-body depth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import torch
from torch import Tensor
from torch.utils.data import Dataset


class TopBodyDepthDataset(Dataset[dict[str, Tensor]]):
    """Load image annotations and return tensors for the regression model.

    An annotation contains normalized ``keypoints_2d`` (which are retained in
    the JSON for visualization), relative depth targets, and per-keypoint
    confidence weights.  Geometric augmentations are intentionally limited to
    small shifts/scales/rotations so depth ordering is not changed.
    """

    def __init__(
        self,
        annotations: str | Path | list[dict[str, Any]],
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        self.root = Path.cwd()
        if isinstance(annotations, (str, Path)):
            annotation_path = Path(annotations)
            self.root = annotation_path.parent.parent if annotation_path.parent.name == "data" else Path.cwd()
            with annotation_path.open("r", encoding="utf-8") as file:
                self.records: list[dict[str, Any]] = json.load(file)
        else:
            self.records = annotations
        if not self.records:
            raise ValueError("No annotations were found.")
        self.image_size = image_size
        self.transform = self._build_transform(augment, image_size)

    @staticmethod
    def _build_transform(augment: bool, image_size: int) -> A.Compose:
        if not augment:
            return A.Compose([A.Resize(image_size, image_size)])
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.ShiftScaleRotate(
                    shift_limit=0.03,
                    scale_limit=0.05,
                    rotate_limit=5,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.5,
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = self.root / image_path
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().div(255.0)
        image_tensor = (image_tensor - 0.5) / 0.5
        depth = torch.tensor(record["relative_depth_gt"], dtype=torch.float32)
        weights = torch.tensor(record.get("confidence_weights", [1.0] * len(depth)), dtype=torch.float32)
        if depth.numel() != weights.numel():
            raise ValueError(f"Depth/weight length mismatch at annotation {index}.")
        return {"image": image_tensor, "depth_gt": depth, "mask": weights}
