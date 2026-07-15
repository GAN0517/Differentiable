from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from Difference.models import fuse_trend_outputs, trend_to_next_state
from Difference.physics.core import PhysicsConfig, physics_tendency
from Difference.physics.grid import SphericalGrid


@dataclass(frozen=True)
class LossConfig:
    mse_weight: float = 1.0
    nonneg_weight: float = 0.0
    sparsity_weight: float = 1e-4
    aod_task_weight: float = 10.0
    t2m_task_weight: float = 1.0
    ceres_task_weight: float = 1.0
    rh_task_weight: float = 1.0
    physics_adv_weight: float = 0.1
    physics_div_weight: float = 0.1
    physics_diff_weight: float = 0.1


FORCE_ZERO_DL_OUTPUT = False


AUX_U10_IDX = 0
AUX_V10_IDX = 1
AUX_T2M_IDX = 2
AUX_CERES_IDX = 3
AUX_RH_IDX = 4
AUX_DIM = 5


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(pred) & torch.isfinite(target)
    if valid.any():
        return F.mse_loss(pred[valid], target[valid])
    return pred.sum() * 0.0


def _masked_weighted_mse(residual: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    valid = torch.isfinite(residual) & torch.isfinite(weight)
    if not valid.any():
        return residual.sum() * 0.0
    w = torch.clamp(weight[valid], min=0.0)
    r = residual[valid]
    denom = torch.sum(w)
    if float(denom.detach().item()) <= 1e-12:
        return torch.mean(r * r)
    return torch.sum(w * r * r) / denom


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
        raise ValueError("D5 is pure-DL. prediction_mode='delta' is disabled.")
    else:
        # D5 baseline: pure DL in trend_sum-style head.
        pred = dl_effective

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
    physics_forward_mode: str = "pure_dl",
    physics_loss_normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    cfg = cfg or LossConfig()

    if feature_seq.ndim != 5:
        raise ValueError(f"feature_seq must be [B,K,Lat,Lon,F], got {tuple(feature_seq.shape)}")
    if target_state_seq.ndim != 5:
        raise ValueError(f"target_state_seq must be [B,K,Lat,Lon,1], got {tuple(target_state_seq.shape)}")
    if not (0.0 < time_decay <= 1.0):
        raise ValueError(f"time_decay must be in (0, 1], got {time_decay}")
    if physics_forward_mode not in {"pure_dl", "coupled"}:
        raise ValueError(
            f"Unsupported physics_forward_mode={physics_forward_mode}. "
            "Use one of: pure_dl, coupled."
        )

    _, steps, _, _, feat_dim = feature_seq.shape
    if not (0 <= int(aod_feature_index) < int(feat_dim)):
        raise ValueError(f"Invalid aod_feature_index={aod_feature_index}, feature dim={feat_dim}")

    # D5PL baseline: keep pure-DL forward, add physics constraints in loss only.
    if bool(online_physics_coupling):
        raise ValueError("D5PL keeps pure-DL rollout. online_physics_coupling must be False.")
    if physics_trend_seq is not None:
        raise ValueError("D5PL physics constraints are computed online in loss. physics_trend_seq must be None.")
    if diffusion_task_model is not None:
        raise ValueError("D5PL does not use diffusion_task_model in forward/loss.")
    if bool(learn_diffusion_coeff):
        raise ValueError("D5PL currently supports constant-K diffusion only. learn_diffusion_coeff must be False.")

    phys_adv_w = float(cfg.physics_adv_weight)
    phys_div_w = float(cfg.physics_div_weight)
    phys_diff_w = float(cfg.physics_diff_weight)
    enable_phys_constraints = (phys_adv_w > 0.0) or (phys_div_w > 0.0) or (phys_diff_w > 0.0)

    if enable_phys_constraints:
        if physics_grid is None or physics_cfg is None:
            raise ValueError(
                "Physics constraints enabled, but physics_grid/physics_cfg is missing. "
                "Pass latitude/longitude grid and physics config from train script."
            )
        if not bool(physics_cfg.enable_diffusion):
            phys_diff_w = 0.0
        if u10_feature_index is None or v10_feature_index is None:
            raise ValueError("Physics constraints enabled, but U10/V10 feature indices are missing.")
        if not (0 <= int(u10_feature_index) < int(feat_dim)):
            raise ValueError(f"Invalid u10_feature_index={u10_feature_index}, feature dim={feat_dim}")
        if not (0 <= int(v10_feature_index) < int(feat_dim)):
            raise ValueError(f"Invalid v10_feature_index={v10_feature_index}, feature dim={feat_dim}")

    # D5 policy: no auxiliary-task supervision.
    if auxiliary_state_seq is not None:
        if auxiliary_state_seq.ndim != 5 or int(auxiliary_state_seq.shape[-1]) != int(AUX_DIM):
            raise ValueError(
                f"auxiliary_state_seq must be [B,K,Lat,Lon,{AUX_DIM}], got {tuple(auxiliary_state_seq.shape)}"
            )
        if auxiliary_state_seq.shape[:4] != target_state_seq.shape[:4]:
            raise ValueError(
                f"auxiliary_state_seq shape prefix must match target_state_seq, got {tuple(auxiliary_state_seq.shape)} vs {tuple(target_state_seq.shape)}"
            )

    aod_prev = feature_seq[:, 0, :, :, aod_feature_index : aod_feature_index + 1]

    pred_states: list[torch.Tensor] = []
    step_losses: list[torch.Tensor] = []
    step_weights: list[float] = []

    step_mse_vals: list[float] = []
    step_nonneg_vals: list[float] = []
    step_sparsity_vals: list[float] = []
    step_pred_state_means: list[float] = []
    step_target_state_means: list[float] = []

    step_phys_means: list[float] = []
    step_dl_means: list[float] = []
    step_total_means: list[float] = []
    step_rad_means: list[float] = []
    step_dep_means: list[float] = []
    step_k_dep_means: list[float] = []
    step_kappa_means: list[float] = []
    step_phys_adv_loss_vals: list[float] = []
    step_phys_div_loss_vals: list[float] = []
    step_phys_diff_loss_vals: list[float] = []
    step_phys_residual_loss_vals: list[float] = []

    step_t2m_loss_vals: list[float] = []
    step_ceres_loss_vals: list[float] = []
    step_rh_loss_vals: list[float] = []

    for s in range(steps):
        weight_s = float(time_decay ** s)
        x_s = feature_seq[:, s, :, :, :].clone()
        x_s[..., aod_feature_index] = aod_prev[..., 0]

        if FORCE_ZERO_DL_OUTPUT:
            trend_dl = torch.zeros_like(aod_prev)
        else:
            trend_dl = model(x_s)
            if trend_dl.shape != aod_prev.shape:
                raise ValueError(
                    f"D5 trend output shape mismatch at step {s+1}: expected {tuple(aod_prev.shape)}, got {tuple(trend_dl.shape)}"
                )

        trend_adv = torch.zeros_like(trend_dl)
        trend_div = torch.zeros_like(trend_dl)
        trend_diff = torch.zeros_like(trend_dl)
        if enable_phys_constraints:
            u_step = x_s[:, :, :, int(u10_feature_index) : int(u10_feature_index) + 1]
            v_step = x_s[:, :, :, int(v10_feature_index) : int(v10_feature_index) + 1]
            tend = physics_tendency(
                aod_prev[..., 0],
                u_step[..., 0],
                v_step[..., 0],
                physics_grid,
                cfg=physics_cfg,
            )
            trend_adv = tend["tendency_adv"].unsqueeze(-1)
            trend_div = tend["tendency_compression"].unsqueeze(-1)
            trend_diff = tend["tendency_diffusion"].unsqueeze(-1)

        trend_phys = trend_adv + trend_div + trend_diff
        aod_next_phys = F.relu(aod_prev + float(dt_days) * trend_phys)
        if physics_forward_mode == "coupled":
            trend_total = trend_dl + trend_phys
        else:
            trend_total = trend_dl
        aod_next = F.relu(aod_prev + float(dt_days) * trend_total)

        target_next = target_state_seq[:, s, :, :, :]
        loss_state_s, parts_s = compute_supervised_loss(aod_next, target_next, cfg=cfg)

        phys_adv_loss_s = trend_dl.sum() * 0.0
        phys_div_loss_s = trend_dl.sum() * 0.0
        phys_diff_loss_s = trend_dl.sum() * 0.0
        phys_loss_s = trend_dl.sum() * 0.0
        if enable_phys_constraints:
            # Physics-loss baseline on next-state consistency:
            #   A_{t+1}^{DL} = A_t + dt * T_DL
            #   A_{t+1}^{phys} = A_t + dt * T_phys
            #   L_phys ~ || A_{t+1}^{DL} - A_{t+1}^{phys} ||^2
            residual = aod_next - aod_next_phys
            if bool(physics_loss_normalize):
                # Normalize by state-change energy to reduce scale mismatch across regimes.
                valid_res = torch.isfinite(residual) & torch.isfinite(target_next)
                if valid_res.any():
                    scale_ref = (target_next - aod_prev)[valid_res]
                    scale = torch.sqrt(torch.mean(scale_ref * scale_ref)).detach()
                    scale = torch.clamp(scale, min=1e-6)
                else:
                    scale = torch.as_tensor(1.0, dtype=residual.dtype, device=residual.device)
                residual = residual / scale

            residual_mse = _masked_mse(residual, torch.zeros_like(residual))
            phys_loss_w = phys_adv_w + phys_div_w + phys_diff_w
            phys_loss_s = phys_loss_w * residual_mse
            # Keep compatibility metric keys; they now represent same residual term.
            phys_adv_loss_s = residual_mse
            phys_div_loss_s = residual_mse
            phys_diff_loss_s = residual_mse

        valid_mask = torch.isfinite(target_next)
        if valid_mask.any():
            sparsity_s = trend_dl[valid_mask].abs().mean()
            pred_state_mean_s = float(aod_next[valid_mask].mean().detach().item())
            target_state_mean_s = float(target_next[valid_mask].mean().detach().item())
            dl_mean_s = float(trend_dl[valid_mask].mean().detach().item())
            total_mean_s = float(trend_total[valid_mask].mean().detach().item())
            phys_mean_s = float(trend_phys[valid_mask].mean().detach().item())
        else:
            sparsity_s = trend_dl.sum() * 0.0
            pred_state_mean_s = 0.0
            target_state_mean_s = 0.0
            dl_mean_s = 0.0
            total_mean_s = 0.0
            phys_mean_s = 0.0

        loss_s = (
            float(cfg.aod_task_weight) * loss_state_s
            + float(cfg.sparsity_weight) * sparsity_s
            + phys_loss_s
        )

        pred_states.append(aod_next)
        step_losses.append(loss_s)
        step_weights.append(weight_s)

        step_mse_vals.append(parts_s["mse"])
        step_nonneg_vals.append(parts_s["nonneg_penalty"])
        step_t2m_loss_vals.append(0.0)
        step_ceres_loss_vals.append(0.0)
        step_rh_loss_vals.append(0.0)

        step_sparsity_vals.append(float(sparsity_s.detach().item()))
        step_pred_state_means.append(pred_state_mean_s)
        step_target_state_means.append(target_state_mean_s)

        step_phys_means.append(phys_mean_s)
        step_dl_means.append(dl_mean_s)
        step_total_means.append(total_mean_s)
        step_rad_means.append(dl_mean_s)
        step_dep_means.append(0.0)
        step_k_dep_means.append(0.0)
        step_kappa_means.append(float("nan"))
        step_phys_adv_loss_vals.append(float(phys_adv_loss_s.detach().item()))
        step_phys_div_loss_vals.append(float(phys_div_loss_s.detach().item()))
        step_phys_diff_loss_vals.append(float(phys_diff_loss_s.detach().item()))
        step_phys_residual_loss_vals.append(float(phys_loss_s.detach().item()))

        aod_prev = aod_next

    weights = torch.as_tensor(step_weights, dtype=step_losses[0].dtype, device=step_losses[0].device)
    loss_vec = torch.stack(step_losses)
    total_loss = torch.sum(loss_vec * weights) / torch.sum(weights)
    pred_state_seq = torch.stack(pred_states, dim=1)

    finite_kappa = [v for v in step_kappa_means if math.isfinite(v)]
    parts = {
        "mse": float(sum(step_mse_vals) / max(len(step_mse_vals), 1)),
        "nonneg_penalty": float(sum(step_nonneg_vals) / max(len(step_nonneg_vals), 1)),
        "t2m_loss": 0.0,
        "ceres_loss": 0.0,
        "rh_loss": 0.0,
        "mse_step_means": list(step_mse_vals),
        "nonneg_step_means": list(step_nonneg_vals),
        "t2m_loss_step_means": list(step_t2m_loss_vals),
        "ceres_loss_step_means": list(step_ceres_loss_vals),
        "rh_loss_step_means": list(step_rh_loss_vals),
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
        "physics_adv_loss_mean": float(sum(step_phys_adv_loss_vals) / max(len(step_phys_adv_loss_vals), 1)),
        "physics_div_loss_mean": float(sum(step_phys_div_loss_vals) / max(len(step_phys_div_loss_vals), 1)),
        "physics_diff_loss_mean": float(sum(step_phys_diff_loss_vals) / max(len(step_phys_diff_loss_vals), 1)),
        "physics_adv_loss_step_means": list(step_phys_adv_loss_vals),
        "physics_div_loss_step_means": list(step_phys_div_loss_vals),
        "physics_diff_loss_step_means": list(step_phys_diff_loss_vals),
        "physics_residual_loss_mean": float(sum(step_phys_residual_loss_vals) / max(len(step_phys_residual_loss_vals), 1)),
        "physics_residual_loss_step_means": list(step_phys_residual_loss_vals),
        "physics_forward_mode": str(physics_forward_mode),
        "physics_loss_normalize": float(1.0 if physics_loss_normalize else 0.0),
        "pred_state_step_means": step_pred_state_means,
        "target_state_step_means": step_target_state_means,
        "rollout_steps": float(steps),
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

    # D5 has no auxiliary state heads; keep signature compatibility only.
    if return_aux_predictions:
        pass
    return total_loss, pred_state_seq, parts

















