from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from Difference.models import fuse_trend_outputs, trend_to_next_state
from Difference.physics.core import PhysicsConfig, physics_tendency
from Difference.physics.grid import SphericalGrid
from Difference.physics.operators import spherical_advection


@dataclass(frozen=True)
class LossConfig:
    mse_weight: float = 1.0
    nonneg_weight: float = 0.0
    sparsity_weight: float = 1e-4
    aod_task_weight: float = 10.0
    u10_task_weight: float = 1.0
    v10_task_weight: float = 1.0
    t2m_task_weight: float = 1.0
    ceres_task_weight: float = 1.0
    rh_task_weight: float = 1.0
    ws_task_weight: float = 1.0
    sp_task_weight: float = 1.0
    blh_task_weight: float = 1.0
    clw_task_weight: float = 1.0


FORCE_ZERO_DL_OUTPUT = False


AUX_U10_IDX = 0
AUX_V10_IDX = 1
AUX_T2M_IDX = 2
AUX_CERES_IDX = 3
AUX_RH_IDX = 4
AUX_WS_IDX = 5
AUX_SP_IDX = 6
AUX_BLH_IDX = 7
AUX_CLW_IDX = 8
AUX_DIM = 9
DIFFUSION_CONTRAST_WINDOW = 5


def combine_physics_and_dl(aod_phys: torch.Tensor, delta_aod_dl: torch.Tensor) -> torch.Tensor:
    return F.relu(aod_phys + delta_aod_dl)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if valid.any():
        return F.mse_loss(pred[valid], target[valid])
    return pred.sum() * 0.0


def _corr_from_sums(
    *,
    count: float,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if not math.isfinite(float(count)) or float(count) <= 1.0:
        return float("nan")
    c = float(count)
    num = float(sum_xy) - (float(sum_x) * float(sum_y)) / c
    var_x = float(sum_x2) - (float(sum_x) * float(sum_x)) / c
    var_y = float(sum_y2) - (float(sum_y) * float(sum_y)) / c
    if var_x <= 1e-12 or var_y <= 1e-12:
        return float("nan")
    return float(num / math.sqrt(var_x * var_y))


def compute_supervised_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    cfg: LossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = cfg or LossConfig()
    mse = _masked_mse(pred, target)
    nonneg_penalty = F.relu(-pred).mean()
    total = cfg.mse_weight * mse + cfg.nonneg_weight * nonneg_penalty
    parts = {
        "mse": float(mse.detach().item()),
        "nonneg_penalty": float(nonneg_penalty.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, parts


def compute_coupled_loss(
    dl_output: torch.Tensor,
    target: torch.Tensor,
    *,
    aod_phys: torch.Tensor | None,
    cfg: LossConfig | None = None,
    prediction_mode: str = "direct",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if prediction_mode not in {"direct", "delta", "trend_sum"}:
        raise ValueError(f"Unsupported prediction_mode: {prediction_mode}")

    dl_effective = torch.zeros_like(dl_output) if FORCE_ZERO_DL_OUTPUT else dl_output

    if prediction_mode == "direct":
        pred = dl_effective
    elif prediction_mode == "delta":
        if aod_phys is None:
            raise ValueError("aod_phys is required when prediction_mode='delta'.")
        pred = combine_physics_and_dl(aod_phys, dl_effective)
    else:
        if aod_phys is None:
            raise ValueError("physics trend is required when prediction_mode='trend_sum'.")
        pred = fuse_trend_outputs(aod_phys, dl_effective)

    total, parts = compute_supervised_loss(pred, target, cfg=cfg)
    return total, pred, parts


def _forward_components_chunked(
    model,
    x: torch.Tensor,
    max_points_per_forward: int,
    *,
    forward_kwargs: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    forward_kwargs = forward_kwargs or {}
    if x.ndim != 4:
        return model.forward_components(x, **forward_kwargs)

    flat_in = x.reshape(-1, x.shape[-1])

    flat_kwargs: dict[str, object] = {}
    x_prefix = tuple(int(v) for v in x.shape[:-1])
    for key, val in forward_kwargs.items():
        if isinstance(val, torch.Tensor) and val.ndim == x.ndim and tuple(int(v) for v in val.shape[:-1]) == x_prefix:
            flat_kwargs[key] = val.reshape(-1, val.shape[-1])
        else:
            flat_kwargs[key] = val

    def _slice_kwargs(start: int, end: int) -> dict[str, object]:
        part_kwargs: dict[str, object] = {}
        for key, val in flat_kwargs.items():
            if isinstance(val, torch.Tensor) and val.ndim >= 2 and int(val.shape[0]) == int(flat_in.shape[0]):
                part_kwargs[key] = val[start:end]
            else:
                part_kwargs[key] = val
        return part_kwargs

    n = int(flat_in.shape[0])
    if max_points_per_forward is None or max_points_per_forward <= 0 or n <= max_points_per_forward:
        flat_out = model.forward_components(flat_in, **_slice_kwargs(0, n))
    else:
        bucket: dict[str, list[torch.Tensor]] = {}
        for start in range(0, n, max_points_per_forward):
            end = min(start + max_points_per_forward, n)
            part = model.forward_components(flat_in[start:end], **_slice_kwargs(start, end))
            for key, val in part.items():
                bucket.setdefault(key, []).append(val)
        flat_out = {key: torch.cat(vals, dim=0) for key, vals in bucket.items()}

    out: dict[str, torch.Tensor] = {}
    for key, flat_val in flat_out.items():
        out[key] = flat_val.reshape(*x.shape[:-1], flat_val.shape[-1])
    return out

def _compute_diffusion_aux_features(
    *,
    aod_prev: torch.Tensor,
    u_prev: torch.Tensor,
    v_prev: torch.Tensor,
    physics_grid: SphericalGrid,
    contrast_window: int = DIFFUSION_CONTRAST_WINDOW,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Online auxiliary features for diffusion-coefficient prediction:
    - aod_advection: metric advection tendency scaled to day^-1
    - aod_local_contrast: scalar minus neighbourhood mean (windowed)
    """
    if int(contrast_window) <= 0 or int(contrast_window) % 2 == 0:
        raise ValueError(f"contrast_window must be positive odd, got {contrast_window}")

    aod = aod_prev[..., 0]
    u = u_prev[..., 0]
    v = v_prev[..., 0]

    # Keep feature semantics aligned with physical_feature_engineering.py.
    aod_adv = spherical_advection(aod, u, v, physics_grid) * 86400.0
    aod_adv = torch.nan_to_num(aod_adv, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)

    half = int(contrast_window) // 2
    aod_4d = aod.unsqueeze(1)  # [B,1,Lat,Lon]
    # Longitude periodic wrap + latitude edge replicate.
    pad_lon = F.pad(aod_4d, (half, half, 0, 0), mode="circular")
    pad_full = F.pad(pad_lon, (0, 0, half, half), mode="replicate")
    neigh_mean = F.avg_pool2d(pad_full, kernel_size=int(contrast_window), stride=1)
    aod_con = torch.nan_to_num(aod_4d - neigh_mean, nan=0.0, posinf=0.0, neginf=0.0)
    aod_con = aod_con.squeeze(1).unsqueeze(-1)
    return aod_adv, aod_con


def _compute_online_physics_step(
    *,
    aod_prev: torch.Tensor,
    u_prev: torch.Tensor,
    v_prev: torch.Tensor,
    ws_prev: torch.Tensor,
    physics_grid: SphericalGrid,
    physics_cfg: PhysicsConfig,
    diffusion_task_model: nn.Module | None,
    max_points_per_forward: int,
    diffusion_k_min: float,
    diffusion_k_base: float,
    diffusion_k_slope: float,
    diffusion_k_cap: float,
    learn_diffusion_coeff: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    # All inputs are [B,Lat,Lon,1]
    k_eff: torch.Tensor | None = None
    if diffusion_task_model is not None:
        aod_adv_feat, aod_con_feat = _compute_diffusion_aux_features(
            aod_prev=aod_prev,
            u_prev=u_prev,
            v_prev=v_prev,
            physics_grid=physics_grid,
            contrast_window=DIFFUSION_CONTRAST_WINDOW,
        )
        diff_in = torch.cat([u_prev, v_prev, ws_prev, aod_adv_feat, aod_con_feat], dim=-1)
        d_out = _forward_components_chunked(
            diffusion_task_model,
            diff_in,
            max_points_per_forward=max_points_per_forward,
        )
        u_next_pred = u_prev
        v_next_pred = v_prev

        if bool(learn_diffusion_coeff):
            kappa_raw = d_out["kappa_raw"]
            ws_mag = torch.abs(ws_prev)
            k_upper = float(diffusion_k_base) + float(diffusion_k_slope) * ws_mag
            k_upper = torch.clamp(
                k_upper,
                min=float(diffusion_k_min) + 1e-12,
                max=max(float(diffusion_k_cap), float(diffusion_k_min) + 1e-12),
            )
            k_eff_full = float(diffusion_k_min) + (k_upper - float(diffusion_k_min)) * torch.sigmoid(kappa_raw)
            k_eff = torch.clamp(k_eff_full[..., 0], min=float(diffusion_k_min))
    else:
        u_next_pred = u_prev
        v_next_pred = v_prev

    tend = physics_tendency(
        aod_prev[..., 0],
        u_prev[..., 0],
        v_prev[..., 0],
        physics_grid,
        cfg=physics_cfg,
        k_eff=k_eff,
    )
    trend_phys = tend["tendency_total"].unsqueeze(-1)
    return trend_phys, u_next_pred, v_next_pred, k_eff


def compute_autoregressive_trend_loss(
    *,
    model,
    feature_seq: torch.Tensor,
    target_state_seq: torch.Tensor,
    physics_trend_seq: torch.Tensor | None,
    auxiliary_state_seq: torch.Tensor | None,
    aod_feature_index: int,
    u10_feature_index: int | None,
    v10_feature_index: int | None,
    ws_feature_index: int | None,
    t2m_feature_index: int | None,
    ceres_feature_index: int | None,
    rh_feature_index: int | None,
    sp_feature_index: int | None,
    blh_feature_index: int | None,
    clw_feature_index: int | None,
    t2m_anom_feature_index: int | None = None,
    ceres_anom_feature_index: int | None = None,
    rh_anom_feature_index: int | None = None,
    dt_days: float = 1.0,
    time_decay: float = 0.9,
    max_points_per_forward: int = 0,
    cfg: LossConfig | None = None,
    online_physics_coupling: bool = False,
    physics_grid: SphericalGrid | None = None,
    physics_cfg: PhysicsConfig | None = None,
    diffusion_task_model: nn.Module | None = None,
    diffusion_k_min: float = 0.0,
    diffusion_k_base: float = 1e-5,
    diffusion_k_slope: float = 5e-6,
    diffusion_k_cap: float = 5e-4,
    iterative_aux_forcing: bool = True,
    learn_diffusion_coeff: bool = False,
    return_aux_predictions: bool = False,
    return_process_predictions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    cfg = cfg or LossConfig()

    if feature_seq.ndim != 5:
        raise ValueError(f"feature_seq must be [B,K,Lat,Lon,F], got {tuple(feature_seq.shape)}")
    if target_state_seq.ndim != 5:
        raise ValueError(f"target_state_seq must be [B,K,Lat,Lon,1], got {tuple(target_state_seq.shape)}")
    if not (0.0 < time_decay <= 1.0):
        raise ValueError(f"time_decay must be in (0, 1], got {time_decay}")

    _, steps, _, _, feat_dim = feature_seq.shape
    if not (0 <= int(aod_feature_index) < int(feat_dim)):
        raise ValueError(f"Invalid aod_feature_index={aod_feature_index}, feature dim={feat_dim}")

    required_idx = {
        "u10_feature_index": u10_feature_index,
        "v10_feature_index": v10_feature_index,
        "ws_feature_index": ws_feature_index,
        "t2m_feature_index": t2m_feature_index,
        "ceres_feature_index": ceres_feature_index,
        "rh_feature_index": rh_feature_index,
        "sp_feature_index": sp_feature_index,
        "blh_feature_index": blh_feature_index,
        "clw_feature_index": clw_feature_index,
    }
    for name, idx in required_idx.items():
        if idx is None or not (0 <= int(idx) < int(feat_dim)):
            raise ValueError(f"Invalid {name}={idx}, feature dim={feat_dim}")

    if bool(online_physics_coupling):
        if physics_grid is None or physics_cfg is None:
            raise ValueError("online_physics_coupling=True requires physics_grid and physics_cfg.")
    else:
        if physics_trend_seq is None:
            raise ValueError("physics_trend_seq is required when online_physics_coupling=False")
        if physics_trend_seq.ndim != 5:
            raise ValueError(f"physics_trend_seq must be [B,K,Lat,Lon,1], got {tuple(physics_trend_seq.shape)}")

    if auxiliary_state_seq is not None:
        if auxiliary_state_seq.ndim != 5 or int(auxiliary_state_seq.shape[-1]) != int(AUX_DIM):
            raise ValueError(
                f"auxiliary_state_seq must be [B,K,Lat,Lon,{AUX_DIM}], got {tuple(auxiliary_state_seq.shape)}"
            )
        if auxiliary_state_seq.shape[:4] != target_state_seq.shape[:4]:
            raise ValueError(
                f"auxiliary_state_seq shape prefix must match target_state_seq, got {tuple(auxiliary_state_seq.shape)} vs {tuple(target_state_seq.shape)}"
            )

    # Current predicted states for iterative rollout.
    aod_prev = feature_seq[:, 0, :, :, aod_feature_index : aod_feature_index + 1]
    u_prev = feature_seq[:, 0, :, :, int(u10_feature_index) : int(u10_feature_index) + 1]
    v_prev = feature_seq[:, 0, :, :, int(v10_feature_index) : int(v10_feature_index) + 1]
    t2m_prev = feature_seq[:, 0, :, :, int(t2m_feature_index) : int(t2m_feature_index) + 1]
    ceres_prev = feature_seq[:, 0, :, :, int(ceres_feature_index) : int(ceres_feature_index) + 1]
    rh_prev = feature_seq[:, 0, :, :, int(rh_feature_index) : int(rh_feature_index) + 1]
    sp_prev = feature_seq[:, 0, :, :, int(sp_feature_index) : int(sp_feature_index) + 1]
    blh_prev = feature_seq[:, 0, :, :, int(blh_feature_index) : int(blh_feature_index) + 1]
    clw_prev = feature_seq[:, 0, :, :, int(clw_feature_index) : int(clw_feature_index) + 1]

    t2m_annual_mean = None
    if t2m_anom_feature_index is not None and 0 <= int(t2m_anom_feature_index) < int(feat_dim):
        t2m_anchor_anom = feature_seq[:, 0, :, :, int(t2m_anom_feature_index) : int(t2m_anom_feature_index) + 1]
        t2m_annual_mean = t2m_prev - t2m_anchor_anom

    ceres_annual_mean = None
    if ceres_anom_feature_index is not None and 0 <= int(ceres_anom_feature_index) < int(feat_dim):
        ceres_anchor_anom = feature_seq[:, 0, :, :, int(ceres_anom_feature_index) : int(ceres_anom_feature_index) + 1]
        ceres_annual_mean = ceres_prev - ceres_anchor_anom

    rh_annual_mean = None
    if rh_anom_feature_index is not None and 0 <= int(rh_anom_feature_index) < int(feat_dim):
        rh_anchor_anom = feature_seq[:, 0, :, :, int(rh_anom_feature_index) : int(rh_anom_feature_index) + 1]
        rh_annual_mean = rh_prev - rh_anchor_anom

    ws_prev = torch.sqrt(torch.clamp(u_prev * u_prev + v_prev * v_prev, min=0.0))

    pred_states = []
    aux_u10_pred_states: list[torch.Tensor] = []
    aux_v10_pred_states: list[torch.Tensor] = []
    aux_t2m_pred_states: list[torch.Tensor] = []
    aux_ceres_pred_states: list[torch.Tensor] = []
    aux_rh_pred_states: list[torch.Tensor] = []
    aux_ws_pred_states: list[torch.Tensor] = []
    aux_sp_pred_states: list[torch.Tensor] = []
    aux_blh_pred_states: list[torch.Tensor] = []
    aux_clw_pred_states: list[torch.Tensor] = []
    proc_src_pred_states: list[torch.Tensor] = []
    proc_dep_pred_states: list[torch.Tensor] = []
    proc_dl_pred_states: list[torch.Tensor] = []
    proc_phys_pred_states: list[torch.Tensor] = []
    proc_aod_phys_pred_states: list[torch.Tensor] = []
    proc_k_dep_pred_states: list[torch.Tensor] = []
    proc_aod_adv_pred_states: list[torch.Tensor] = []
    proc_aod_con_pred_states: list[torch.Tensor] = []
    proc_u10_forcing_states: list[torch.Tensor] = []
    proc_v10_forcing_states: list[torch.Tensor] = []
    proc_ws_forcing_states: list[torch.Tensor] = []
    proc_t2m_forcing_states: list[torch.Tensor] = []
    proc_ceres_forcing_states: list[torch.Tensor] = []
    proc_rh_forcing_states: list[torch.Tensor] = []
    proc_sp_forcing_states: list[torch.Tensor] = []
    proc_blh_forcing_states: list[torch.Tensor] = []
    proc_clw_forcing_states: list[torch.Tensor] = []
    proc_t2m_anom_states: list[torch.Tensor] = []
    proc_ceres_anom_states: list[torch.Tensor] = []
    proc_rh_anom_states: list[torch.Tensor] = []
    proc_t2m_ann_mean_states: list[torch.Tensor] = []
    proc_ceres_ann_mean_states: list[torch.Tensor] = []
    proc_rh_ann_mean_states: list[torch.Tensor] = []
    step_losses = []
    step_weights = []

    step_mse_vals = []
    step_nonneg_vals = []
    step_u10_loss_vals = []
    step_v10_loss_vals = []
    step_t2m_loss_vals = []
    step_ceres_loss_vals = []
    step_rh_loss_vals = []
    step_ws_loss_vals = []
    step_sp_loss_vals = []
    step_blh_loss_vals = []
    step_clw_loss_vals = []

    step_phys_means = []
    step_dl_means = []
    step_total_means = []
    step_rad_means = []
    step_dep_means = []
    step_k_dep_means = []
    step_kappa_means = []
    step_sparsity_vals = []
    step_pred_state_means = []
    step_target_state_means = []
    src_dep_count = 0.0
    src_dep_sum_src = 0.0
    src_dep_sum_dep = 0.0
    src_dep_sum_src2 = 0.0
    src_dep_sum_dep2 = 0.0
    src_dep_sum_cross = 0.0

    for s in range(steps):
        weight_s = float(time_decay ** s)

        x_s = feature_seq[:, s, :, :, :].clone()
        x_s[..., aod_feature_index] = aod_prev[..., 0]

        if bool(iterative_aux_forcing):
            u_step = u_prev
            v_step = v_prev
            ws_step = ws_prev
            t2m_step = t2m_prev
            ceres_step = ceres_prev
            rh_step = rh_prev
            sp_step = sp_prev
            blh_step = blh_prev
            clw_step = clw_prev
        else:
            # Dallforcing1 fixed-forcing policy: always use anchor-time atmospheric background (step 0).
            u_step = feature_seq[:, 0, :, :, int(u10_feature_index) : int(u10_feature_index) + 1]
            v_step = feature_seq[:, 0, :, :, int(v10_feature_index) : int(v10_feature_index) + 1]
            ws_step = feature_seq[:, 0, :, :, int(ws_feature_index) : int(ws_feature_index) + 1]
            t2m_step = feature_seq[:, 0, :, :, int(t2m_feature_index) : int(t2m_feature_index) + 1]
            ceres_step = feature_seq[:, 0, :, :, int(ceres_feature_index) : int(ceres_feature_index) + 1]
            rh_step = feature_seq[:, 0, :, :, int(rh_feature_index) : int(rh_feature_index) + 1]
            sp_step = feature_seq[:, 0, :, :, int(sp_feature_index) : int(sp_feature_index) + 1]
            blh_step = feature_seq[:, 0, :, :, int(blh_feature_index) : int(blh_feature_index) + 1]
            clw_step = feature_seq[:, 0, :, :, int(clw_feature_index) : int(clw_feature_index) + 1]

        x_s[..., int(u10_feature_index)] = u_step[..., 0]
        x_s[..., int(v10_feature_index)] = v_step[..., 0]
        x_s[..., int(ws_feature_index)] = ws_step[..., 0]
        x_s[..., int(t2m_feature_index)] = t2m_step[..., 0]
        x_s[..., int(ceres_feature_index)] = ceres_step[..., 0]
        x_s[..., int(rh_feature_index)] = rh_step[..., 0]
        x_s[..., int(sp_feature_index)] = sp_step[..., 0]
        x_s[..., int(blh_feature_index)] = blh_step[..., 0]
        x_s[..., int(clw_feature_index)] = clw_step[..., 0]

        if t2m_annual_mean is not None and t2m_anom_feature_index is not None:
            x_s[..., int(t2m_anom_feature_index)] = (t2m_step - t2m_annual_mean)[..., 0]
        if ceres_annual_mean is not None and ceres_anom_feature_index is not None:
            x_s[..., int(ceres_anom_feature_index)] = (ceres_step - ceres_annual_mean)[..., 0]
        if rh_annual_mean is not None and rh_anom_feature_index is not None:
            x_s[..., int(rh_anom_feature_index)] = (rh_step - rh_annual_mean)[..., 0]

        t2m_anom_step = (
            x_s[..., int(t2m_anom_feature_index) : int(t2m_anom_feature_index) + 1]
            if t2m_anom_feature_index is not None
            else torch.zeros_like(aod_prev)
        )
        ceres_anom_step = (
            x_s[..., int(ceres_anom_feature_index) : int(ceres_anom_feature_index) + 1]
            if ceres_anom_feature_index is not None
            else torch.zeros_like(aod_prev)
        )
        rh_anom_step = (
            x_s[..., int(rh_anom_feature_index) : int(rh_anom_feature_index) + 1]
            if rh_anom_feature_index is not None
            else torch.zeros_like(aod_prev)
        )

        if physics_grid is not None:
            proc_aod_adv, proc_aod_con = _compute_diffusion_aux_features(
                aod_prev=aod_prev,
                u_prev=u_step,
                v_prev=v_step,
                physics_grid=physics_grid,
                contrast_window=DIFFUSION_CONTRAST_WINDOW,
            )
        else:
            proc_aod_adv = torch.zeros_like(aod_prev)
            proc_aod_con = torch.zeros_like(aod_prev)

        if bool(online_physics_coupling):
            trend_phys, u_next_pred_phy, v_next_pred_phy, k_eff_s = _compute_online_physics_step(
                aod_prev=aod_prev,
                u_prev=u_step,
                v_prev=v_step,
                ws_prev=ws_step,
                physics_grid=physics_grid,
                physics_cfg=physics_cfg,
                diffusion_task_model=diffusion_task_model,
                max_points_per_forward=max_points_per_forward,
                diffusion_k_min=float(diffusion_k_min),
                diffusion_k_base=float(diffusion_k_base),
                diffusion_k_slope=float(diffusion_k_slope),
                diffusion_k_cap=float(diffusion_k_cap),
                learn_diffusion_coeff=bool(learn_diffusion_coeff),
            )
        else:
            assert physics_trend_seq is not None
            trend_phys = physics_trend_seq[:, s, :, :, :]
            u_next_pred_phy = u_step
            v_next_pred_phy = v_step
            k_eff_s = None

        aod_phy = trend_to_next_state(aod_prev, trend_phys, dt_days=dt_days, enforce_nonneg=True)

        if FORCE_ZERO_DL_OUTPUT:
            trend_rad = torch.zeros_like(trend_phys)
            k_dep = torch.zeros_like(trend_phys)
            u10_next_pred = u_step
            v10_next_pred = v_step
            ws_next_pred = ws_step
            t2m_next_pred = t2m_step
            ceres_next_pred = ceres_step
            rh_next_pred = rh_step
            sp_next_pred = sp_step
            blh_next_pred = blh_step
            clw_next_pred = clw_step
        else:
            comps = _forward_components_chunked(
                model,
                x_s,
                max_points_per_forward=max_points_per_forward,
                forward_kwargs={
                    "aod_adv": proc_aod_adv,
                    "aod_con": proc_aod_con,
                },
            )
            trend_rad = comps["trend_rad"]
            k_dep = comps["k_dep"]
            u10_next_pred = comps["u10_next"]
            v10_next_pred = comps["v10_next"]
            ws_next_pred = comps["ws_next"]
            t2m_next_pred = comps["t2m_next"]
            ceres_next_pred = comps["ceres_next"]
            rh_next_pred = comps["rh_next"]
            sp_next_pred = comps["sp_next"]
            blh_next_pred = comps["blh_next"]
            clw_next_pred = comps["clw_next"]

        trend_dep = -k_dep * aod_phy
        trend_dl = trend_rad + trend_dep
        trend_total = fuse_trend_outputs(trend_phys, trend_dl)
        aod_next = F.relu(aod_phy + dt_days * trend_dl)

        target_next = target_state_seq[:, s, :, :, :]
        loss_state_s, parts_s = compute_supervised_loss(aod_next, target_next, cfg=cfg)

        u10_loss = loss_state_s * 0.0
        v10_loss = loss_state_s * 0.0
        t2m_loss = loss_state_s * 0.0
        ceres_loss = loss_state_s * 0.0
        rh_loss = loss_state_s * 0.0
        ws_loss = loss_state_s * 0.0
        sp_loss = loss_state_s * 0.0
        blh_loss = loss_state_s * 0.0
        clw_loss = loss_state_s * 0.0

        if auxiliary_state_seq is not None:
            aux_next = auxiliary_state_seq[:, s, :, :, :]
            u10_true_next = aux_next[..., AUX_U10_IDX : AUX_U10_IDX + 1]
            v10_true_next = aux_next[..., AUX_V10_IDX : AUX_V10_IDX + 1]
            t2m_true_next = aux_next[..., AUX_T2M_IDX : AUX_T2M_IDX + 1]
            ceres_true_next = aux_next[..., AUX_CERES_IDX : AUX_CERES_IDX + 1]
            rh_true_next = aux_next[..., AUX_RH_IDX : AUX_RH_IDX + 1]
            ws_true_next = aux_next[..., AUX_WS_IDX : AUX_WS_IDX + 1]
            sp_true_next = aux_next[..., AUX_SP_IDX : AUX_SP_IDX + 1]
            blh_true_next = aux_next[..., AUX_BLH_IDX : AUX_BLH_IDX + 1]
            clw_true_next = aux_next[..., AUX_CLW_IDX : AUX_CLW_IDX + 1]
            u10_loss = _masked_mse(u10_next_pred, u10_true_next)
            v10_loss = _masked_mse(v10_next_pred, v10_true_next)
            t2m_loss = _masked_mse(t2m_next_pred, t2m_true_next)
            ceres_loss = _masked_mse(ceres_next_pred, ceres_true_next)
            rh_loss = _masked_mse(rh_next_pred, rh_true_next)
            ws_loss = _masked_mse(ws_next_pred, ws_true_next)
            sp_loss = _masked_mse(sp_next_pred, sp_true_next)
            blh_loss = _masked_mse(blh_next_pred, blh_true_next)
            clw_loss = _masked_mse(clw_next_pred, clw_true_next)

        valid_mask = torch.isfinite(target_next)
        if valid_mask.any():
            sparsity_s = trend_rad[valid_mask].abs().mean() + trend_dep[valid_mask].abs().mean()
            pred_state_mean_s = float(aod_next[valid_mask].mean().detach().item())
            target_state_mean_s = float(target_next[valid_mask].mean().detach().item())
            phys_mean_s = float(trend_phys[valid_mask].mean().detach().item())
            dl_mean_s = float(trend_dl[valid_mask].mean().detach().item())
            total_mean_s = float(trend_total[valid_mask].mean().detach().item())
            rad_mean_s = float(trend_rad[valid_mask].mean().detach().item())
            dep_mean_s = float(trend_dep[valid_mask].mean().detach().item())
            k_dep_mean_s = float(k_dep[valid_mask].mean().detach().item())
            if k_eff_s is not None:
                vm2 = valid_mask[..., 0]
                kappa_mean_s = float(k_eff_s[vm2].mean().detach().item()) if vm2.any() else float("nan")
            else:
                kappa_mean_s = float("nan")
            # Robust global src/dep correlation accumulation.
            # Use masked weighted sums instead of boolean advanced indexing to
            # avoid potential backend instability on some CUDA stacks.
            valid_w = valid_mask.to(dtype=trend_rad.dtype)
            src_safe = torch.nan_to_num(trend_rad, nan=0.0, posinf=0.0, neginf=0.0)
            dep_safe = torch.nan_to_num(trend_dep, nan=0.0, posinf=0.0, neginf=0.0)
            src_dep_count += float(torch.sum(valid_w).detach().item())
            src_dep_sum_src += float(torch.sum(src_safe * valid_w).detach().item())
            src_dep_sum_dep += float(torch.sum(dep_safe * valid_w).detach().item())
            src_dep_sum_src2 += float(torch.sum(src_safe * src_safe * valid_w).detach().item())
            src_dep_sum_dep2 += float(torch.sum(dep_safe * dep_safe * valid_w).detach().item())
            src_dep_sum_cross += float(torch.sum(src_safe * dep_safe * valid_w).detach().item())
        else:
            sparsity_s = trend_rad.sum() * 0.0
            pred_state_mean_s = 0.0
            target_state_mean_s = 0.0
            phys_mean_s = 0.0
            dl_mean_s = 0.0
            total_mean_s = 0.0
            rad_mean_s = 0.0
            dep_mean_s = 0.0
            k_dep_mean_s = 0.0
            kappa_mean_s = float("nan")

        loss_s = (
            float(cfg.aod_task_weight) * loss_state_s
            + float(cfg.u10_task_weight) * u10_loss
            + float(cfg.v10_task_weight) * v10_loss
            + float(cfg.t2m_task_weight) * t2m_loss
            + float(cfg.ceres_task_weight) * ceres_loss
            + float(cfg.rh_task_weight) * rh_loss
            + float(cfg.ws_task_weight) * ws_loss
            + float(cfg.sp_task_weight) * sp_loss
            + float(cfg.blh_task_weight) * blh_loss
            + float(cfg.clw_task_weight) * clw_loss
            + float(cfg.sparsity_weight) * sparsity_s
        )

        pred_states.append(aod_next)
        if return_aux_predictions:
            aux_u10_pred_states.append(u10_next_pred)
            aux_v10_pred_states.append(v10_next_pred)
            aux_t2m_pred_states.append(t2m_next_pred)
            aux_ceres_pred_states.append(ceres_next_pred)
            aux_rh_pred_states.append(rh_next_pred)
            aux_ws_pred_states.append(ws_next_pred)
            aux_sp_pred_states.append(sp_next_pred)
            aux_blh_pred_states.append(blh_next_pred)
            aux_clw_pred_states.append(clw_next_pred)
        if return_process_predictions:
            proc_src_pred_states.append(trend_rad)
            proc_dep_pred_states.append(trend_dep)
            proc_dl_pred_states.append(trend_dl)
            proc_phys_pred_states.append(trend_phys)
            proc_aod_phys_pred_states.append(aod_phy)
            proc_k_dep_pred_states.append(k_dep)
            proc_aod_adv_pred_states.append(proc_aod_adv)
            proc_aod_con_pred_states.append(proc_aod_con)
            proc_u10_forcing_states.append(u_step)
            proc_v10_forcing_states.append(v_step)
            proc_ws_forcing_states.append(ws_step)
            proc_t2m_forcing_states.append(t2m_step)
            proc_ceres_forcing_states.append(ceres_step)
            proc_rh_forcing_states.append(rh_step)
            proc_sp_forcing_states.append(sp_step)
            proc_blh_forcing_states.append(blh_step)
            proc_clw_forcing_states.append(clw_step)
            proc_t2m_anom_states.append(t2m_anom_step)
            proc_ceres_anom_states.append(ceres_anom_step)
            proc_rh_anom_states.append(rh_anom_step)
            proc_t2m_ann_mean_states.append(t2m_annual_mean if t2m_annual_mean is not None else torch.zeros_like(aod_prev))
            proc_ceres_ann_mean_states.append(ceres_annual_mean if ceres_annual_mean is not None else torch.zeros_like(aod_prev))
            proc_rh_ann_mean_states.append(rh_annual_mean if rh_annual_mean is not None else torch.zeros_like(aod_prev))
        step_losses.append(loss_s)
        step_weights.append(weight_s)

        step_mse_vals.append(parts_s["mse"])
        step_nonneg_vals.append(parts_s["nonneg_penalty"])
        step_u10_loss_vals.append(float(u10_loss.detach().item()))
        step_v10_loss_vals.append(float(v10_loss.detach().item()))
        step_t2m_loss_vals.append(float(t2m_loss.detach().item()))
        step_ceres_loss_vals.append(float(ceres_loss.detach().item()))
        step_rh_loss_vals.append(float(rh_loss.detach().item()))
        step_ws_loss_vals.append(float(ws_loss.detach().item()))
        step_sp_loss_vals.append(float(sp_loss.detach().item()))
        step_blh_loss_vals.append(float(blh_loss.detach().item()))
        step_clw_loss_vals.append(float(clw_loss.detach().item()))

        step_sparsity_vals.append(float(sparsity_s.detach().item()))
        step_phys_means.append(phys_mean_s)
        step_dl_means.append(dl_mean_s)
        step_total_means.append(total_mean_s)
        step_rad_means.append(rad_mean_s)
        step_dep_means.append(dep_mean_s)
        step_k_dep_means.append(k_dep_mean_s)
        step_kappa_means.append(kappa_mean_s)
        step_pred_state_means.append(pred_state_mean_s)
        step_target_state_means.append(target_state_mean_s)

        # Iterative state update.
        aod_prev = aod_next

        if bool(iterative_aux_forcing):
            u_prev = u10_next_pred
            v_prev = v10_next_pred
            ws_prev = ws_next_pred
            t2m_prev = t2m_next_pred
            ceres_prev = ceres_next_pred
            rh_prev = rh_next_pred
            sp_prev = sp_next_pred
            blh_prev = blh_next_pred
            clw_prev = clw_next_pred
        else:
            u_prev = u_step
            v_prev = v_step
            ws_prev = ws_step
            t2m_prev = t2m_step
            ceres_prev = ceres_step
            rh_prev = rh_step
            sp_prev = sp_step
            blh_prev = blh_step
            clw_prev = clw_step

    weights = torch.as_tensor(step_weights, dtype=step_losses[0].dtype, device=step_losses[0].device)
    loss_vec = torch.stack(step_losses)
    total_loss = torch.sum(loss_vec * weights) / torch.sum(weights)
    pred_state_seq = torch.stack(pred_states, dim=1)

    finite_kappa = [v for v in step_kappa_means if math.isfinite(v)]
    src_dep_corr = _corr_from_sums(
        count=src_dep_count,
        sum_x=src_dep_sum_src,
        sum_y=src_dep_sum_dep,
        sum_x2=src_dep_sum_src2,
        sum_y2=src_dep_sum_dep2,
        sum_xy=src_dep_sum_cross,
    )

    parts = {
        "mse": float(sum(step_mse_vals) / max(len(step_mse_vals), 1)),
        "nonneg_penalty": float(sum(step_nonneg_vals) / max(len(step_nonneg_vals), 1)),
        "u10_loss": float(sum(step_u10_loss_vals) / max(len(step_u10_loss_vals), 1)),
        "v10_loss": float(sum(step_v10_loss_vals) / max(len(step_v10_loss_vals), 1)),
        "t2m_loss": float(sum(step_t2m_loss_vals) / max(len(step_t2m_loss_vals), 1)),
        "ceres_loss": float(sum(step_ceres_loss_vals) / max(len(step_ceres_loss_vals), 1)),
        "rh_loss": float(sum(step_rh_loss_vals) / max(len(step_rh_loss_vals), 1)),
        "ws_loss": float(sum(step_ws_loss_vals) / max(len(step_ws_loss_vals), 1)),
        "sp_loss": float(sum(step_sp_loss_vals) / max(len(step_sp_loss_vals), 1)),
        "blh_loss": float(sum(step_blh_loss_vals) / max(len(step_blh_loss_vals), 1)),
        "clw_loss": float(sum(step_clw_loss_vals) / max(len(step_clw_loss_vals), 1)),
        "mse_step_means": list(step_mse_vals),
        "nonneg_step_means": list(step_nonneg_vals),
        "u10_loss_step_means": list(step_u10_loss_vals),
        "v10_loss_step_means": list(step_v10_loss_vals),
        "t2m_loss_step_means": list(step_t2m_loss_vals),
        "ceres_loss_step_means": list(step_ceres_loss_vals),
        "rh_loss_step_means": list(step_rh_loss_vals),
        "ws_loss_step_means": list(step_ws_loss_vals),
        "sp_loss_step_means": list(step_sp_loss_vals),
        "blh_loss_step_means": list(step_blh_loss_vals),
        "clw_loss_step_means": list(step_clw_loss_vals),
        "sparsity_penalty": float(sum(step_sparsity_vals) / max(len(step_sparsity_vals), 1)),
        "sparsity_step_means": list(step_sparsity_vals),
        "total": float(total_loss.detach().item()),
        "time_decay": float(time_decay),
        "phys_trend_mean": float(sum(step_phys_means) / max(len(step_phys_means), 1)),
        "dl_trend_mean": float(sum(step_dl_means) / max(len(step_dl_means), 1)),
        "total_trend_mean": float(sum(step_total_means) / max(len(step_total_means), 1)),
        "rad_trend_mean": float(sum(step_rad_means) / max(len(step_rad_means), 1)),
        "dep_trend_mean": float(sum(step_dep_means) / max(len(step_dep_means), 1)),
        "k_dep_mean": float(sum(step_k_dep_means) / max(len(step_k_dep_means), 1)),
        "kappa_diff_mean": float(sum(finite_kappa) / max(len(finite_kappa), 1)) if finite_kappa else float("nan"),
        "phys_trend_step_means": step_phys_means,
        "dl_trend_step_means": step_dl_means,
        "total_trend_step_means": step_total_means,
        "rad_trend_step_means": step_rad_means,
        "dep_trend_step_means": step_dep_means,
        "k_dep_step_means": step_k_dep_means,
        "kappa_diff_step_means": step_kappa_means,
        "pred_state_step_means": step_pred_state_means,
        "target_state_step_means": step_target_state_means,
        "rollout_steps": float(steps),
        "src_dep_corr": float(src_dep_corr),
        "src_dep_count": float(src_dep_count),
        "src_dep_sum_src": float(src_dep_sum_src),
        "src_dep_sum_dep": float(src_dep_sum_dep),
        "src_dep_sum_src2": float(src_dep_sum_src2),
        "src_dep_sum_dep2": float(src_dep_sum_dep2),
        "src_dep_sum_cross": float(src_dep_sum_cross),
        # backward-compatible aliases
        "dry_trend_mean": float(sum(step_dep_means) / max(len(step_dep_means), 1)),
        "wet_trend_mean": 0.0,
        "k_dry_mean": float(sum(step_k_dep_means) / max(len(step_k_dep_means), 1)),
        "k_wet_mean": 0.0,
        "dry_trend_step_means": step_dep_means,
        "wet_trend_step_means": [0.0 for _ in step_dep_means],
        "k_dry_step_means": step_k_dep_means,
        "k_wet_step_means": [0.0 for _ in step_k_dep_means],
    }

    if return_aux_predictions:
        parts["pred_u10_state_seq"] = torch.stack(aux_u10_pred_states, dim=1)
        parts["pred_v10_state_seq"] = torch.stack(aux_v10_pred_states, dim=1)
        parts["pred_t2m_state_seq"] = torch.stack(aux_t2m_pred_states, dim=1)
        parts["pred_ceres_state_seq"] = torch.stack(aux_ceres_pred_states, dim=1)
        parts["pred_rh_state_seq"] = torch.stack(aux_rh_pred_states, dim=1)
        parts["pred_ws_state_seq"] = torch.stack(aux_ws_pred_states, dim=1)
        parts["pred_sp_state_seq"] = torch.stack(aux_sp_pred_states, dim=1)
        parts["pred_blh_state_seq"] = torch.stack(aux_blh_pred_states, dim=1)
        parts["pred_clw_state_seq"] = torch.stack(aux_clw_pred_states, dim=1)
    if return_process_predictions:
        parts["pred_trend_src_seq"] = torch.stack(proc_src_pred_states, dim=1)
        parts["pred_trend_dep_seq"] = torch.stack(proc_dep_pred_states, dim=1)
        parts["pred_trend_dl_seq"] = torch.stack(proc_dl_pred_states, dim=1)
        parts["pred_trend_phys_seq"] = torch.stack(proc_phys_pred_states, dim=1)
        parts["pred_aod_phys_state_seq"] = torch.stack(proc_aod_phys_pred_states, dim=1)
        parts["pred_k_dep_seq"] = torch.stack(proc_k_dep_pred_states, dim=1)
        parts["pred_aod_adv_seq"] = torch.stack(proc_aod_adv_pred_states, dim=1)
        parts["pred_aod_con_seq"] = torch.stack(proc_aod_con_pred_states, dim=1)
        parts["forcing_u10_state_seq"] = torch.stack(proc_u10_forcing_states, dim=1)
        parts["forcing_v10_state_seq"] = torch.stack(proc_v10_forcing_states, dim=1)
        parts["forcing_ws_state_seq"] = torch.stack(proc_ws_forcing_states, dim=1)
        parts["forcing_t2m_state_seq"] = torch.stack(proc_t2m_forcing_states, dim=1)
        parts["forcing_ceres_state_seq"] = torch.stack(proc_ceres_forcing_states, dim=1)
        parts["forcing_rh_state_seq"] = torch.stack(proc_rh_forcing_states, dim=1)
        parts["forcing_sp_state_seq"] = torch.stack(proc_sp_forcing_states, dim=1)
        parts["forcing_blh_state_seq"] = torch.stack(proc_blh_forcing_states, dim=1)
        parts["forcing_clw_state_seq"] = torch.stack(proc_clw_forcing_states, dim=1)
        parts["forcing_t2m_anom_seq"] = torch.stack(proc_t2m_anom_states, dim=1)
        parts["forcing_ceres_anom_seq"] = torch.stack(proc_ceres_anom_states, dim=1)
        parts["forcing_rh_anom_seq"] = torch.stack(proc_rh_anom_states, dim=1)
        parts["forcing_t2m_annual_mean_seq"] = torch.stack(proc_t2m_ann_mean_states, dim=1)
        parts["forcing_ceres_annual_mean_seq"] = torch.stack(proc_ceres_ann_mean_states, dim=1)
        parts["forcing_rh_annual_mean_seq"] = torch.stack(proc_rh_ann_mean_states, dim=1)
    return total_loss, pred_state_seq, parts

















