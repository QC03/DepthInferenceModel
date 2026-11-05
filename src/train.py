"""Train and validate the relative-depth estimator."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from dataset import TopBodyDepthDataset
from model import ResNet18RelativeDepthEstimator

BONE_EDGES = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (1, 7), (2, 8), (7, 8)]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _bone_lengths(keypoints_2d: Tensor, depth: Tensor) -> Tensor:
    """Return one normalized 3D length for each configured skeleton edge."""
    xyz = torch.cat((keypoints_2d, depth.unsqueeze(-1)), dim=-1)
    return torch.stack([torch.linalg.vector_norm(xyz[:, start] - xyz[:, end], dim=-1) for start, end in BONE_EDGES], dim=-1)


def compute_bone_bounds(records: list[dict[str, object]], margin: float = 0.15) -> tuple[Tensor, Tensor]:
    """Estimate per-bone min/max lengths from labels with a safety margin."""
    lengths: list[Tensor] = []
    for record in records:
        points = torch.tensor(record["keypoints_2d"], dtype=torch.float32)
        depth = torch.tensor(record["relative_depth_gt"], dtype=torch.float32)
        lengths.append(_bone_lengths(points.unsqueeze(0), depth.unsqueeze(0)).squeeze(0))
    values = torch.stack(lengths)
    minimum = values.min(dim=0).values * (1.0 - margin)
    maximum = values.max(dim=0).values * (1.0 + margin)
    return minimum, maximum


def weighted_smooth_l1(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    keypoints_2d: Tensor,
    bone_min: Tensor,
    bone_max: Tensor,
    bone_loss_weight: float = 0.2,
) -> Tensor:
    """Combine confidence-weighted depth loss with bone-length range penalties."""
    loss = nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    depth_loss = (loss * weight).sum() / weight.sum().clamp_min(1e-8)
    lengths = _bone_lengths(keypoints_2d, prediction)
    range_penalty = torch.relu(bone_min.to(lengths.device) - lengths).square()
    range_penalty += torch.relu(lengths - bone_max.to(lengths.device)).square()
    return depth_loss + bone_loss_weight * range_penalty.mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    absolute_error = 0.0
    count = 0.0
    for batch in loader:
        prediction = model(batch["image"].to(device))
        target = batch["depth_gt"].to(device)
        weight = batch["mask"].to(device)
        absolute_error += (prediction.sub(target).abs() * weight).sum().item()
        count += weight.sum().item()
    return absolute_error / max(count, 1e-8)


def save_checkpoint(payload: dict[str, Any], checkpoint: Path, epoch: int, retries: int = 5) -> None:
    """Persist a checkpoint while tolerating transient OneDrive/file locks."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    epoch_path = checkpoint.with_name(f"{checkpoint.stem}.epoch_{epoch:03d}{checkpoint.suffix}")
    torch.save(payload, epoch_path)
    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            shutil.copyfile(epoch_path, checkpoint)
            return
        except OSError as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise OSError(
        f"Unable to update checkpoint '{checkpoint}'. "
        f"The per-epoch checkpoint was preserved at '{epoch_path}'. "
        "Close programs syncing or opening the checkpoint and retry."
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations.json"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    args = parser.parse_args()
    seed_everything(args.seed)
    records = json.loads(args.annotations.read_text(encoding="utf-8"))
    if len(records) < 2:
        raise ValueError("At least two annotations are required for an 80/20 split.")
    indices = list(range(len(records)))
    generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(indices), generator=generator).tolist()
    split = max(1, int(len(indices) * 0.8))
    train_indices, validation_indices = permutation[:split], permutation[split:]
    if not validation_indices:
        validation_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    train_set = TopBodyDepthDataset(records, augment=True)
    validation_set = TopBodyDepthDataset(records, augment=False)
    train_loader = DataLoader(Subset(train_set, train_indices), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(validation_set, validation_indices), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18RelativeDepthEstimator(len(records[0]["relative_depth_gt"])).to(device)
    bone_min, bone_max = compute_bone_bounds(records)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_mae = float("inf")
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["image"].to(device))
            loss = weighted_smooth_l1(
                prediction,
                batch["depth_gt"].to(device),
                batch["mask"].to(device),
                batch["keypoints_2d"].to(device),
                bone_min,
                bone_max,
            )
            loss.backward()
            optimizer.step()
        scheduler.step()
        mae = evaluate(model, validation_loader, device)
        print(f"epoch={epoch:03d} validation_mae={mae:.6f}")
        if mae < best_mae:
            best_mae = mae
            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "num_keypoints": len(records[0]["relative_depth_gt"]),
                    "keypoint_names": records[0].get("keypoint_names"),
                    "validation_mae": mae,
                    "bone_edges": BONE_EDGES,
                    "bone_min": bone_min.tolist(),
                    "bone_max": bone_max.tolist(),
                },
                args.checkpoint,
                epoch,
            )
    print(f"Best checkpoint: {args.checkpoint} (MAE={best_mae:.6f})")


if __name__ == "__main__":
    main()
