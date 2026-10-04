"""Pixel preprocessing.

uint8 HWC frames -> ImageNet-normalised float CHW tensors. The dataloader
workers and every analysis script use the same function, so preprocessing never
differs between training and evaluation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageTransform:
    """uint8 (..., H, W, 3) -> float (..., 3, size, size), ImageNet-normalised.

    Args:
        size: output resolution; frames already at `size` skip the resize.
        mean, std: channel statistics.
    """

    def __init__(self, size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(np.ascontiguousarray(frames)) if isinstance(frames, np.ndarray) else frames
        if x.ndim == 3:  # a single frame
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)  # (N, 3, H, W)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Back to uint8 (N, H, W, 3) — for videos and debugging."""
        x = (x.detach().cpu() * self.std + self.mean).clamp(0, 1)
        return (x.permute(0, 2, 3, 1) * 255).to(torch.uint8)
