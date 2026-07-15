from __future__ import annotations

import torch
import torch.nn.functional as F


def fuse_trend_outputs(trend_phys: torch.Tensor, trend_dl: torch.Tensor) -> torch.Tensor:
    """
    Core coupling rule:
    total trend = physics trend + DL trend
    """
    return trend_phys + trend_dl


def trend_to_next_state(
    aod_t: torch.Tensor,
    trend_total: torch.Tensor,
    dt_days: float = 1.0,
    enforce_nonneg: bool = True,
) -> torch.Tensor:
    """
    Optional state update from trend output.
    """
    aod_next = aod_t + dt_days * trend_total
    return F.relu(aod_next) if enforce_nonneg else aod_next

