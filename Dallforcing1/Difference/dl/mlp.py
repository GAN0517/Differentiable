from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 128)
    dropout: float = 0.1
    output_dim: int = 1


@dataclass(frozen=True)
class DiffusionTaskMLPConfig:
    input_dim: int = 5  # [u10, v10, ws, aod_advection, aod_local_contrast]
    hidden_dims: tuple[int, ...] = (64, 64)
    dropout: float = 0.0


@dataclass(frozen=True)
class ProcessDecomposedMLPConfig:
    feature_dim: int
    aod_index: int
    u10_index: int
    v10_index: int
    wind_speed_index: int
    ceres_index: int
    sza_index: int | None
    t2m_index: int
    blh_index: int
    tp_index: int
    clw_index: int
    rh_index: int
    sp_index: int
    ceres_anom_index: int | None = None
    t2m_anom_index: int | None = None
    rh_anom_index: int | None = None
    hidden_dims: tuple[int, ...] = (128, 128)
    dropout: float = 0.1
    rad_init_bias: float = -1.0
    dry_init_bias: float = -3.0
    wet_init_bias: float = -3.0


class GridPointMLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.cfg = cfg
        self.net = _build_mlp(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            output_dim=cfg.output_dim,
            dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"GridPointMLP expects tensor with last feature dim, got shape={tuple(x.shape)}")
        in_shape = x.shape[:-1]
        feat_dim = x.shape[-1]
        x_flat = x.reshape(-1, feat_dim)
        y_flat = self.net(x_flat)
        return y_flat.reshape(*in_shape, self.cfg.output_dim)


class DiffusionTaskMLP(nn.Module):
    """
    Diffusion task branch output:
    - kappa_raw: unconstrained diffusion logit
    """

    def __init__(self, cfg: DiffusionTaskMLPConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = _build_trunk(
            input_dim=cfg.input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        trunk_dim = int(cfg.hidden_dims[-1]) if len(cfg.hidden_dims) > 0 else int(cfg.input_dim)
        self.head_kappa = nn.Linear(trunk_dim, 1)

    def forward_components(
        self,
        x: torch.Tensor,
        *,
        aod_adv: torch.Tensor | None = None,
        aod_con: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim < 2:
            raise ValueError(f"DiffusionTaskMLP expects tensor with last feature dim, got shape={tuple(x.shape)}")
        if int(x.shape[-1]) != int(self.cfg.input_dim):
            raise ValueError(
                f"DiffusionTaskMLP expected feature dim={self.cfg.input_dim}, got {int(x.shape[-1])}"
            )
        in_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        h = self.trunk(x_flat)
        kappa_raw = self.head_kappa(h).reshape(*in_shape, 1)
        return {"kappa_raw": kappa_raw}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_components(x)["kappa_raw"]


class ProcessDecomposedMLP(nn.Module):
    """
    Multi-task process model with shared trunk per process branch:
    - Radiation branch:
        * trend_rad (AOD source term)
        * t2m_next (next-step T2M state)
        * ceres_next (next-step CERES state)
        * sp_next / blh_next / clw_next (next-step forcing states)
    - Deposition branch:
        * k_dep (merged deposition coefficient)
        * u10_next / v10_next / ws_next (next-step wind states)
        * rh_next (next-step RH state)
    """

    def __init__(self, cfg: ProcessDecomposedMLPConfig):
        super().__init__()
        self.cfg = cfg

        self._use_ceres_anom = cfg.ceres_anom_index is not None
        self._use_t2m_anom = cfg.t2m_anom_index is not None
        self._use_rh_anom = cfg.rh_anom_index is not None
        rad_input_dim = 9 + int(self._use_ceres_anom) + int(self._use_t2m_anom)

        # Radiation trunk input: [AOD_prev, U10, V10, WS, CERES, (CERES_ANOM), SZA, T2M, (T2M_ANOM), AODADV, AODCON]
        self.rad_trunk = _build_trunk(
            input_dim=rad_input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        rad_dim = int(cfg.hidden_dims[-1]) if len(cfg.hidden_dims) > 0 else rad_input_dim
        self.head_trend_rad = nn.Linear(rad_dim, 1)
        self.head_t2m_next = nn.Linear(rad_dim, 1)
        self.head_ceres_next = nn.Linear(rad_dim, 1)
        self.head_sp_next = nn.Linear(rad_dim, 1)
        self.head_blh_next = nn.Linear(rad_dim, 1)
        self.head_clw_next = nn.Linear(rad_dim, 1)

        # Deposition trunk input:
        # [AOD_prev, U10, V10, WS, BLH, T2M, TP, CLW, RH, (RH_ANOM), SP, AODADV, AODCON]
        self.dep_trunk = _build_trunk(
            input_dim=12 + int(self._use_rh_anom),
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
        )
        dep_input_dim = 12 + int(self._use_rh_anom)
        dep_dim = int(cfg.hidden_dims[-1]) if len(cfg.hidden_dims) > 0 else dep_input_dim
        self.head_k_dep = nn.Linear(dep_dim, 1)
        self.head_u10_next = nn.Linear(dep_dim, 1)
        self.head_v10_next = nn.Linear(dep_dim, 1)
        self.head_ws_next = nn.Linear(dep_dim, 1)
        self.head_rh_next = nn.Linear(dep_dim, 1)

        _init_linear_bias(self.head_trend_rad, cfg.rad_init_bias)
        _init_linear_bias(self.head_k_dep, cfg.dry_init_bias)

    def _pick(self, x_flat: torch.Tensor, idx: int | None) -> torch.Tensor:
        if idx is None:
            return torch.zeros((x_flat.shape[0], 1), device=x_flat.device, dtype=x_flat.dtype)
        return x_flat[:, idx : idx + 1]

    def forward_components(
        self,
        x: torch.Tensor,
        *,
        aod_adv: torch.Tensor | None = None,
        aod_con: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim < 2:
            raise ValueError(f"ProcessDecomposedMLP expects tensor with last feature dim, got shape={tuple(x.shape)}")
        if x.shape[-1] != self.cfg.feature_dim:
            raise ValueError(
                f"ProcessDecomposedMLP expected feature dim={self.cfg.feature_dim}, got {int(x.shape[-1])}"
            )

        in_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        aod_prev = self._pick(x_flat, self.cfg.aod_index)
        u10 = self._pick(x_flat, self.cfg.u10_index)
        v10 = self._pick(x_flat, self.cfg.v10_index)
        ws = self._pick(x_flat, self.cfg.wind_speed_index)
        ceres = self._pick(x_flat, self.cfg.ceres_index)
        ceres_anom = self._pick(x_flat, self.cfg.ceres_anom_index)
        sza = self._pick(x_flat, self.cfg.sza_index)
        t2m = self._pick(x_flat, self.cfg.t2m_index)
        t2m_anom = self._pick(x_flat, self.cfg.t2m_anom_index)
        blh = self._pick(x_flat, self.cfg.blh_index)
        tp = self._pick(x_flat, self.cfg.tp_index)
        clw = self._pick(x_flat, self.cfg.clw_index)
        rh = self._pick(x_flat, self.cfg.rh_index)
        rh_anom = self._pick(x_flat, self.cfg.rh_anom_index)
        sp = self._pick(x_flat, self.cfg.sp_index)

        if aod_adv is None:
            aod_adv_flat = torch.zeros_like(aod_prev)
        else:
            if aod_adv.shape != (*in_shape, 1):
                raise ValueError(
                    f"aod_adv shape mismatch: expected {(*in_shape, 1)}, got {tuple(aod_adv.shape)}"
                )
            aod_adv_flat = aod_adv.reshape(-1, 1)

        if aod_con is None:
            aod_con_flat = torch.zeros_like(aod_prev)
        else:
            if aod_con.shape != (*in_shape, 1):
                raise ValueError(
                    f"aod_con shape mismatch: expected {(*in_shape, 1)}, got {tuple(aod_con.shape)}"
                )
            aod_con_flat = aod_con.reshape(-1, 1)

        rad_terms = [aod_prev, u10, v10, ws, ceres]
        if self._use_ceres_anom:
            rad_terms.append(ceres_anom)
        rad_terms.extend([sza, t2m])
        if self._use_t2m_anom:
            rad_terms.append(t2m_anom)
        rad_terms.extend([aod_adv_flat, aod_con_flat])
        rad_in = torch.cat(rad_terms, dim=-1)
        dep_terms = [aod_prev, u10, v10, ws, blh, t2m, tp, clw, rh]
        if self._use_rh_anom:
            dep_terms.append(rh_anom)
        dep_terms.append(sp)
        dep_terms.extend([aod_adv_flat, aod_con_flat])
        dep_in = torch.cat(dep_terms, dim=-1)

        rad_h = self.rad_trunk(rad_in)
        dep_h = self.dep_trunk(dep_in)

        trend_rad = self.head_trend_rad(rad_h).reshape(*in_shape, 1)
        t2m_next = self.head_t2m_next(rad_h).reshape(*in_shape, 1)
        ceres_next = self.head_ceres_next(rad_h).reshape(*in_shape, 1)
        sp_next = self.head_sp_next(rad_h).reshape(*in_shape, 1)
        blh_next = self.head_blh_next(rad_h).reshape(*in_shape, 1)
        clw_next = self.head_clw_next(rad_h).reshape(*in_shape, 1)

        k_dep = torch.sigmoid(self.head_k_dep(dep_h)).reshape(*in_shape, 1)
        u10_next = self.head_u10_next(dep_h).reshape(*in_shape, 1)
        v10_next = self.head_v10_next(dep_h).reshape(*in_shape, 1)
        ws_next = self.head_ws_next(dep_h).reshape(*in_shape, 1)
        rh_next = self.head_rh_next(dep_h).reshape(*in_shape, 1)

        aod_ref = aod_prev.reshape(*in_shape, 1)

        return {
            "trend_rad": trend_rad,
            "k_dep": k_dep,
            "u10_next": u10_next,
            "v10_next": v10_next,
            "ws_next": ws_next,
            "t2m_next": t2m_next,
            "ceres_next": ceres_next,
            "rh_next": rh_next,
            "sp_next": sp_next,
            "blh_next": blh_next,
            "clw_next": clw_next,
            "aod_ref": aod_ref,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        comps = self.forward_components(x)
        return comps["trend_rad"] - comps["k_dep"] * comps["aod_ref"]


def _build_trunk(
    input_dim: int,
    hidden_dims: Iterable[int],
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for width in hidden_dims:
        w = int(width)
        layers.append(nn.Linear(prev, w))
        layers.append(nn.ReLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = w
    if not layers:
        return nn.Sequential(nn.Identity())
    return nn.Sequential(*layers)


def _build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = int(input_dim)
    for width in hidden_dims:
        w = int(width)
        layers.append(nn.Linear(prev, w))
        layers.append(nn.ReLU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        prev = w
    layers.append(nn.Linear(prev, int(output_dim)))
    return nn.Sequential(*layers)


def _init_linear_bias(layer: nn.Linear, bias_value: float) -> None:
    nn.init.constant_(layer.bias, float(bias_value))


def _init_last_linear_bias(net: nn.Sequential, bias_value: float) -> None:
    for layer in reversed(list(net)):
        if isinstance(layer, nn.Linear):
            nn.init.constant_(layer.bias, float(bias_value))
            return









