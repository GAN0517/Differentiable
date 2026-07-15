from __future__ import annotations

import torch
import torch.nn.functional as F

from .grid import SphericalGrid


def d_dlambda(field: torch.Tensor, dlon_rad: float) -> torch.Tensor:
    """Central difference in longitude (periodic boundary)."""
    return (torch.roll(field, shifts=-1, dims=-1) - torch.roll(field, shifts=1, dims=-1)) / (2.0 * dlon_rad)


def d_dphi(field: torch.Tensor, dlat_rad: float) -> torch.Tensor:
    """Central difference in latitude with one-sided boundary at poles."""
    grad = torch.empty_like(field)
    grad[..., 1:-1, :] = (field[..., 2:, :] - field[..., :-2, :]) / (2.0 * dlat_rad)
    grad[..., 0, :] = (field[..., 1, :] - field[..., 0, :]) / dlat_rad
    grad[..., -1, :] = (field[..., -1, :] - field[..., -2, :]) / dlat_rad
    return grad


def d2_dlambda2(field: torch.Tensor, dlon_rad: float) -> torch.Tensor:
    """Second derivative in longitude (periodic boundary)."""
    return (
        torch.roll(field, shifts=-1, dims=-1)
        - 2.0 * field
        + torch.roll(field, shifts=1, dims=-1)
    ) / (dlon_rad * dlon_rad)


def spherical_advection(aod: torch.Tensor, u: torch.Tensor, v: torch.Tensor, grid: SphericalGrid) -> torch.Tensor:
    """
    Compute dA/dt advection term on sphere:
    -(u/(R cos(phi)) * dA/dlambda + v/R * dA/dphi)
    """
    dA_dlambda = d_dlambda(aod, grid.dlon_rad)
    dA_dphi = d_dphi(aod, grid.dlat_rad)
    cos_lat = grid.cos_lat.view(1, -1, 1)
    r = grid.earth_radius_m
    return -((u / (r * cos_lat)) * dA_dlambda + (v / r) * dA_dphi)


def spherical_divergence(u: torch.Tensor, v: torch.Tensor, grid: SphericalGrid) -> torch.Tensor:
    """
    Horizontal wind divergence on sphere:
    div = 1/(R cos(phi)) * [du/dlambda + d(v cos(phi))/dphi]
    """
    cos_lat = grid.cos_lat.view(1, -1, 1)
    du_dlambda = d_dlambda(u, grid.dlon_rad)
    d_vcos_dphi = d_dphi(v * cos_lat, grid.dlat_rad)
    return (du_dlambda + d_vcos_dphi) / (grid.earth_radius_m * cos_lat)


def spherical_laplacian(aod: torch.Tensor, grid: SphericalGrid) -> torch.Tensor:
    """
    Scalar Laplacian on sphere for latitude/longitude coordinates.
    """
    cos_lat = grid.cos_lat.view(1, -1, 1)
    r2 = float(grid.earth_radius_m) ** 2
    dA_dphi = d_dphi(aod, grid.dlat_rad)
    term_phi = d_dphi(cos_lat * dA_dphi, grid.dlat_rad) / cos_lat
    term_lon = d2_dlambda2(aod, grid.dlon_rad) / (cos_lat * cos_lat)
    return (term_phi + term_lon) / r2


def spherical_diffusion_tendency(
    aod: torch.Tensor,
    k_eff: torch.Tensor,
    grid: SphericalGrid,
) -> torch.Tensor:
    """
    Compute diffusion tendency on sphere:
        div(K * grad(A))
    where K can be spatially varying.
    """
    if aod.shape != k_eff.shape:
        raise ValueError(
            f"spherical_diffusion_tendency expects aod/k_eff same shape, got {tuple(aod.shape)} vs {tuple(k_eff.shape)}"
        )
    cos_lat = grid.cos_lat.view(1, -1, 1)
    r2 = float(grid.earth_radius_m) ** 2
    dA_dlambda = d_dlambda(aod, grid.dlon_rad)
    dA_dphi = d_dphi(aod, grid.dlat_rad)
    term_lon = d_dlambda(k_eff * dA_dlambda, grid.dlon_rad) / (cos_lat * cos_lat)
    term_phi = d_dphi(cos_lat * k_eff * dA_dphi, grid.dlat_rad) / cos_lat
    return (term_lon + term_phi) / r2


def semi_lagrangian_advect(
    aod: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    grid: SphericalGrid,
    dt_days: float = 1.0,
) -> torch.Tensor:
    """
    Semi-Lagrangian advection using torch.grid_sample on lat-lon grid.
    Returns advected scalar at next time step.
    """
    if aod.ndim != 3:
        raise ValueError(f"semi_lagrangian_advect expects [B,Lat,Lon], got {tuple(aod.shape)}")

    _, lat_n, lon_n = aod.shape
    device = aod.device
    dtype = aod.dtype

    lat = grid.lat_rad.to(device=device, dtype=dtype)
    lon = grid.lon_rad.to(device=device, dtype=dtype)
    lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")  # [Lat, Lon]

    cos_lat = grid.cos_lat.to(device=device, dtype=dtype).clamp_min(1e-6)
    cos_lat_grid = cos_lat[:, None].expand(lat_n, lon_n)
    r = float(grid.earth_radius_m)
    dt_sec = float(dt_days) * 86400.0

    # Back-trajectory departure points.
    dphi = (v * dt_sec) / r
    dlambda = (u * dt_sec) / (r * cos_lat_grid[None, :, :])
    lat_dep = lat_grid[None, :, :] - dphi
    lon_dep = lon_grid[None, :, :] - dlambda

    # Longitude periodic wrap.
    lon_min = float(lon.min().item())
    lon_max = float(lon.max().item())
    dlon = float(grid.dlon_rad)
    period = (lon_max - lon_min) + dlon
    lon_dep = torch.remainder(lon_dep - lon_min, period) + lon_min

    # Latitude clamp to valid domain.
    lat_min = float(lat.min().item())
    lat_max = float(lat.max().item())
    lat_dep = torch.clamp(lat_dep, min=lat_min, max=lat_max)

    # Normalize to grid_sample coordinates [-1, 1].
    x = 2.0 * (lon_dep - lon_min) / max(lon_max - lon_min, 1e-8) - 1.0
    y = 2.0 * (lat_dep - lat_min) / max(lat_max - lat_min, 1e-8) - 1.0
    sample_grid = torch.stack((x, y), dim=-1)  # [B, Lat, Lon, 2]

    aod_in = aod.unsqueeze(1)  # [B,1,Lat,Lon]
    advected = F.grid_sample(
        aod_in,
        sample_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return advected.squeeze(1)
