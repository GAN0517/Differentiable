from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SphericalGrid:
    """Spherical grid metadata for latitude/longitude operators."""

    lat_deg: torch.Tensor
    lon_deg: torch.Tensor
    earth_radius_m: float = 6_371_000.0
    cos_floor: float = 0.05

    @property
    def lat_rad(self) -> torch.Tensor:
        return torch.deg2rad(self.lat_deg)

    @property
    def lon_rad(self) -> torch.Tensor:
        return torch.deg2rad(self.lon_deg)

    @property
    def dlat_rad(self) -> float:
        return _uniform_spacing_rad(self.lat_deg)

    @property
    def dlon_rad(self) -> float:
        return _uniform_spacing_rad(self.lon_deg)

    @property
    def cos_lat(self) -> torch.Tensor:
        cos_lat = torch.cos(self.lat_rad)
        floor = torch.as_tensor(self.cos_floor, dtype=cos_lat.dtype, device=cos_lat.device)
        return torch.clamp(cos_lat, min=floor)

    @property
    def area(self) -> torch.Tensor:
        """Approximate cell area [lat, lon] in m^2."""
        r2 = self.earth_radius_m ** 2
        dphi = self.dlat_rad
        dlambda = self.dlon_rad
        lat = self.lat_rad
        lat_low = lat - 0.5 * dphi
        lat_high = lat + 0.5 * dphi
        strip = r2 * dlambda * (torch.sin(lat_high) - torch.sin(lat_low))
        return strip[:, None].repeat(1, self.lon_deg.numel())


def _uniform_spacing_rad(deg_values: torch.Tensor) -> float:
    if deg_values.numel() < 2:
        raise ValueError("Latitude/longitude coordinate needs at least two points.")
    diffs = torch.diff(deg_values).abs()
    step_deg = float(torch.median(diffs).item())
    if step_deg <= 0.0:
        raise ValueError("Invalid coordinate spacing: non-positive degree step.")
    return float(torch.deg2rad(torch.tensor(step_deg)).item())

