from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FeatureTargetDataset(Dataset):
    """Minimal dataset wrapper for MLP training."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        physics_baseline: np.ndarray | None = None,
        auxiliary_targets: np.ndarray | None = None,
    ):
        if features.shape[0] != targets.shape[0]:
            raise ValueError(
                f"features and targets must share first axis, got {features.shape[0]} vs {targets.shape[0]}"
            )
        if physics_baseline is not None and physics_baseline.shape[0] != features.shape[0]:
            raise ValueError(
                f"physics_baseline and features must share first axis, got {physics_baseline.shape[0]} vs {features.shape[0]}"
            )
        if auxiliary_targets is not None and auxiliary_targets.shape[0] != features.shape[0]:
            raise ValueError(
                "auxiliary_targets and features must share first axis, "
                f"got {auxiliary_targets.shape[0]} vs {features.shape[0]}"
            )
        self.x = torch.from_numpy(features.astype(np.float32))
        self.y = torch.from_numpy(targets.astype(np.float32))
        self.p = None if physics_baseline is None else torch.from_numpy(physics_baseline.astype(np.float32))
        self.aux = None if auxiliary_targets is None else torch.from_numpy(auxiliary_targets.astype(np.float32))

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.p is None and self.aux is None:
            return self.x[idx], self.y[idx]
        if self.p is None and self.aux is not None:
            return self.x[idx], self.y[idx], self.aux[idx]
        if self.p is not None and self.aux is None:
            return self.x[idx], self.y[idx], self.p[idx]
        return self.x[idx], self.y[idx], self.p[idx], self.aux[idx]
