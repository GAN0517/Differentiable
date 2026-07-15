from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .grid import SphericalGrid
from .operators import (
    semi_lagrangian_advect,
    spherical_advection,
    spherical_diffusion_tendency,
    spherical_divergence,
)


@dataclass(frozen=True)
class PhysicsConfig:
    """Hyperparameters for differentiable AOD physics update."""

    dt_days: float = 1.0
    eps: float = 1e-8
    advection_scheme: str = "semi_lagrangian"  # choices: semi_lagrangian, central
    enable_diffusion: bool = False
    diffusion_k_const: float = 0.0


def physics_tendency(
    aod: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: SphericalGrid,
    cfg: PhysicsConfig | None = None,
    *,
    k_eff: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Spherical transport + compressibility tendency (+ optional diffusion).
    dA/dt = adv - A*div + div(K * grad(A))
    """
    cfg = cfg or PhysicsConfig()
    if cfg.advection_scheme == "semi_lagrangian":
        aod_advected = semi_lagrangian_advect(aod, u, v, grid, dt_days=cfg.dt_days)
        adv = (aod_advected - aod) / max(cfg.dt_days, cfg.eps)
    elif cfg.advection_scheme == "central":
        adv = spherical_advection(aod, u, v, grid)
    else:
        raise ValueError(f"Unsupported advection_scheme: {cfg.advection_scheme}")
    div = spherical_divergence(u, v, grid)
    compression = -aod * div

    diffusion = torch.zeros_like(compression)
    if bool(cfg.enable_diffusion):
        if k_eff is None:
            k_eff_use = torch.full_like(aod, float(cfg.diffusion_k_const))
        else:
            if k_eff.shape != aod.shape:
                raise ValueError(
                    f"k_eff shape must match aod shape. got k_eff={tuple(k_eff.shape)} aod={tuple(aod.shape)}"
                )
            k_eff_use = torch.clamp(k_eff, min=0.0)
        diffusion = spherical_diffusion_tendency(aod, k_eff_use, grid)

    total = adv + compression + diffusion
    return {
        "tendency_total": total,
        "tendency_adv": adv,
        "tendency_compression": compression,
        "tendency_diffusion": diffusion,
        "divergence": div,
    }


def step_aod_physics(
    aod_t: torch.Tensor,
    u_t: torch.Tensor,
    v_t: torch.Tensor,
    grid: SphericalGrid,
    cfg: PhysicsConfig | None = None,
    *,
    k_eff: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    One-step differentiable physics update (Euler forward).

    Input tensors use shape [B, Lat, Lon] (or [Lat, Lon], batch optional).
    """
    cfg = cfg or PhysicsConfig()
    tend = physics_tendency(aod_t, u_t, v_t, grid, cfg=cfg, k_eff=k_eff)
    aod_next = aod_t + cfg.dt_days * tend["tendency_total"]
    aod_next = F.relu(aod_next)  # Protocol 6: hard non-negative constraint
    return {
        "aod_next_phys": aod_next,
        **tend,
    }
