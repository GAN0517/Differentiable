from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from Difference.dl import (
    DiffusionTaskMLP,
    DiffusionTaskMLPConfig,
    GridPointMLP,
    MLPConfig,
    ProcessDecomposedMLP,
    ProcessDecomposedMLPConfig,
)
from Difference.physics.core import PhysicsConfig, physics_tendency
from Difference.physics.grid import SphericalGrid
from Difference.physics.operators import spherical_advection
from Difference.settings import GLOBAL_SETTINGS, PHYSICS_SETTINGS, TRAIN_SETTINGS

TIME_DIM = "time"
COMPUTED_FEATURE_KEYS = {"T2M_ANOM", "CERES_ANOM", "R_ANOM"}

FEATURE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "U10": ("u10",),
    "V10": ("v10",),
    "WS": ("wind_speed",),
    "T2M": ("t2m",),
    "BLH": ("blh",),
    "R": ("r",),
    "SP": ("sp",),
    "TP": ("tp",),
    "AOD": ("AODANA",),
    "AODI": ("AODINC",),
    "CERES": ("ceres_all_sky_sw_down",),
    "CERES_ALL": ("ceres_all_sky_sw_down",),
    "CERES_ANOM": ("ceres_all_sky_sw_down_annual_anom",),
    "SZA": ("ceres_sza", "sza", "solar_zenith_angle"),
    "T2M_ANOM": ("t2m_annual_anom",),
    "R_ANOM": ("r_annual_anom",),
    "CLW": ("cloud_liquid_water", "tclw", "clwc"),
}

DEFAULT_FEATURE_KEYS = [
    "U10",
    "V10",
    "WS",
    "T2M",
    "BLH",
    "R",
    "SP",
    "TP",
    "AOD",
    "AODI",
    "CERES_ALL",
    "SZA",
    "CLW",
]

D2_MULTI_TASK_TARGETS = ["aod_src", "aod_dep", "t2m_next", "ceres_next", "rh_next"]
D2_PROCESS_CONTRIB_TARGETS = ["trend_rad", "trend_dep", "trend_total"]
FULL_TREND_TARGET = "full_trend"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D2 SHAP analysis for pure-DL vs process-decomposed models.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path)
    parser.add_argument("--data-nc", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--years", type=int, nargs="+", default=list(TRAIN_SETTINGS.get("years", GLOBAL_SETTINGS["years"])))
    parser.add_argument("--feature-keys", type=str, nargs="+", default=DEFAULT_FEATURE_KEYS)
    parser.add_argument("--require-all-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--target",
        choices=(
            "model_output",
            "aod_src",
            "aod_dep",
            "t2m_next",
            "ceres_next",
            "rh_next",
            "trend_rad",
            "trend_dep",
            "trend_total",
            FULL_TREND_TARGET,
            "k_dep",
            "dep_term",
        ),
        default="model_output",
    )
    parser.add_argument(
        "--all-contrib",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run process contribution SHAP in one shot: "
            "trend_rad (src), trend_dep (dep), trend_total (src+dep)."
        ),
    )
    parser.add_argument(
        "--include-full-trend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When used with --all-contrib, also save full_trend SHAP. "
            "full_trend is computed as a precomputed physics tendency variable plus the DL trend_total term."
        ),
    )
    parser.add_argument(
        "--full-trend-physics-var",
        type=str,
        default="tendency_total",
        help="NetCDF variable used as the physics tendency term for --include-full-trend / --target full_trend.",
    )
    parser.add_argument(
        "--full-trend-physics-source",
        choices=("auto", "var", "online"),
        default="auto",
        help=(
            "Physics tendency source for full_trend. 'var' loads --full-trend-physics-var from NetCDF; "
            "'online' recomputes the differentiable physics core from AOD/U10/V10; "
            "'auto' tries var first and falls back to online."
        ),
    )
    parser.add_argument(
        "--full-trend-physics-feature-name",
        type=str,
        default=None,
        help=(
            "Feature name used in full_trend SHAP outputs for the appended physics tendency column. "
            "Defaults to the value of --full-trend-physics-var."
        ),
    )
    parser.add_argument("--dt-days", type=float, default=float(TRAIN_SETTINGS.get("dt_days", 1.0)))
    parser.add_argument(
        "--physics-advection-scheme",
        choices=("semi_lagrangian", "central"),
        default=str(PHYSICS_SETTINGS.get("scheme", "semi_lagrangian")),
    )
    parser.add_argument(
        "--physics-enable-diffusion",
        action=argparse.BooleanOptionalAction,
        default=bool(PHYSICS_SETTINGS.get("enable_diffusion", False)),
    )
    parser.add_argument(
        "--physics-diffusion-k-const",
        type=float,
        default=float(PHYSICS_SETTINGS.get("diffusion_k_const", 0.0)),
    )
    parser.add_argument(
        "--physics-cos-floor",
        type=float,
        default=float(PHYSICS_SETTINGS.get("cos_floor", 0.05)),
    )
    parser.add_argument(
        "--full-trend-online-batch-days",
        type=int,
        default=8,
        help="Number of time slices per chunk when recomputing online physics tendency for full_trend.",
    )
    parser.add_argument(
        "--all-tasks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run all D2 process-task SHAP analyses in one shot: "
            "AOD src/dep + T2M/CERES/RH auxiliary tasks. "
            "This excludes model_output/physics terms."
        ),
    )
    parser.add_argument("--sample-step", type=int, default=0, help="When input is [N,K,...,F], choose step index.")
    parser.add_argument("--background-size", type=int, default=128)
    parser.add_argument(
        "--explain-size",
        type=int,
        default=256,
        help="Number of explained samples. <=0 means use all samples after background selection.",
    )
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument(
        "--append-annual-anomalies",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether to append annual-anomaly channels (T2M/CERES/RH) for NetCDF inputs. "
            "If omitted, the script auto-selects based on checkpoint input_dim."
        ),
    )
    parser.add_argument(
        "--explain-batch-size",
        type=int,
        default=2048,
        help="Chunk size when computing SHAP values to control memory usage.",
    )
    parser.add_argument("--explainer", choices=("gradient", "kernel"), default="gradient")
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("Difference/analysis/shap_outputs"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def resolve_data_paths(base: Path, years: list[int] | None, filename: str) -> list[Path]:
    if not years:
        p = base / filename
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")
        return [p]
    out: list[Path] = []
    for y in years:
        cands = [base / str(int(y)) / filename, base / f"D{int(y)}" / filename]
        found = next((c for c in cands if c.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Required file not found for year={y}. Tried: {', '.join(str(c) for c in cands)}")
        out.append(found)
    return out


def resolve_feature_names(ds: xr.Dataset, feature_keys: list[str], require_all: bool) -> list[str]:
    names: list[str] = []
    missing: list[str] = []
    for key in feature_keys:
        key_up = key.upper()
        if key_up in COMPUTED_FEATURE_KEYS:
            continue
        cands = FEATURE_CANDIDATES.get(key_up)
        if cands is None:
            missing.append(f"{key} (unknown key)")
            continue
        chosen = next((n for n in cands if n in ds.data_vars), None)
        if chosen is None:
            missing.append(f"{key} -> {cands}")
            continue
        names.append(chosen)
    if missing and require_all:
        raise KeyError("Missing required features: " + "; ".join(missing))
    if missing:
        logging.warning("Skipping missing features: %s", "; ".join(missing))
    return names


def _first_resolved_feature(feature_names: list[str], key: str) -> str | None:
    candidates = FEATURE_CANDIDATES.get(key.upper(), ())
    return next((name for name in feature_names if name in candidates), None)


def _annual_anomaly_feature(ds: xr.Dataset, var_name: str, out_name: str) -> xr.DataArray:
    da = ds[var_name]
    if TIME_DIM not in da.dims:
        raise ValueError(f"Cannot build annual anomaly for {var_name}; missing '{TIME_DIM}' dimension.")
    anom = da.groupby(f"{TIME_DIM}.year") - da.groupby(f"{TIME_DIM}.year").mean(TIME_DIM, skipna=True)
    anom = anom.astype(np.float32)
    anom.name = out_name
    return anom


def _feature_array_with_annual_anomalies(ds: xr.Dataset, feature_names: list[str]) -> tuple[xr.DataArray, list[str]]:
    arrays: list[xr.DataArray] = [ds[name] for name in feature_names]
    names = list(feature_names)
    additions: list[tuple[str, str, str]] = [
        ("T2M", "t2m_annual_anom", "T2M annual anomaly"),
        ("CERES_ALL", "ceres_all_sky_sw_down_annual_anom", "CERES annual anomaly"),
        ("R", "r_annual_anom", "RH annual anomaly"),
    ]
    for key, out_name, label in additions:
        base_name = _first_resolved_feature(names, key)
        if base_name is None:
            logging.warning("Skipping %s feature because base feature key '%s' was not resolved.", label, key)
            continue
        if out_name in names:
            continue
        arrays.append(_annual_anomaly_feature(ds, base_name, out_name))
        names.append(out_name)
    feat_da = xr.concat(arrays, dim=xr.IndexVariable("feature", names))
    return feat_da.transpose("latitude", "longitude", TIME_DIM, "feature"), names


def load_feature_array_from_nc(
    *,
    data_nc: Path | None,
    data_root: Path | None,
    years: list[int] | None,
    feature_keys: list[str],
    require_all_features: bool,
    append_annual_anomalies: bool | None,
    expected_input_dim: int | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if data_nc is not None:
        if not data_nc.exists():
            raise FileNotFoundError(f"NetCDF not found: {data_nc}")
        paths = [data_nc]
    else:
        if data_root is None:
            data_root = Path(GLOBAL_SETTINGS["data_root"])
        filename = str(TRAIN_SETTINGS.get("trend_input_filename", TRAIN_SETTINGS.get("input_filename", "data_with_physics.nc")))
        paths = resolve_data_paths(data_root, years, filename)
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords", engine="netcdf4")
    try:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": TIME_DIM})
        if "number" in ds.coords or "number" in ds:
            ds = ds.drop_vars("number", errors="ignore")
        feat_names = resolve_feature_names(ds, feature_keys, require_all=require_all_features)
        if "pressure_level" in ds.dims:
            ds = ds.isel(pressure_level=0).drop_vars("pressure_level", errors="ignore")
        if "level" in ds.dims:
            ds = ds.isel(level=0).drop_vars("level", errors="ignore")
        # Auto-select anomaly-channel usage to match checkpoint input dimension.
        # This keeps one SHAP script compatible with both:
        # - no-anomaly models (e.g., Dallforcing1 anchor-bg)
        # - anomaly-enabled models.
        if append_annual_anomalies is True:
            feat_da, feat_names = _feature_array_with_annual_anomalies(ds, feat_names)
            logging.info("Feature mode: with annual anomalies (forced by CLI).")
        elif append_annual_anomalies is False:
            feat_da = xr.concat([ds[name] for name in feat_names], dim=xr.IndexVariable("feature", feat_names))
            feat_da = feat_da.transpose("latitude", "longitude", TIME_DIM, "feature")
            logging.info("Feature mode: base features only (forced by CLI).")
        else:
            feat_da_base = xr.concat([ds[name] for name in feat_names], dim=xr.IndexVariable("feature", feat_names))
            feat_da_base = feat_da_base.transpose("latitude", "longitude", TIME_DIM, "feature")
            base_dim = int(len(feat_names))
            feat_da_anom, feat_names_anom = _feature_array_with_annual_anomalies(ds, feat_names)
            anom_dim = int(len(feat_names_anom))

            if expected_input_dim is None:
                feat_da, feat_names = feat_da_anom, feat_names_anom
                logging.info(
                    "Feature mode: with annual anomalies (auto, no expected_input_dim provided). dim=%d",
                    anom_dim,
                )
            elif int(expected_input_dim) == base_dim:
                feat_da, feat_names = feat_da_base, feat_names
                logging.info(
                    "Feature mode: base features only (auto by ckpt input_dim=%d).",
                    int(expected_input_dim),
                )
            elif int(expected_input_dim) == anom_dim:
                feat_da, feat_names = feat_da_anom, feat_names_anom
                logging.info(
                    "Feature mode: with annual anomalies (auto by ckpt input_dim=%d).",
                    int(expected_input_dim),
                )
            else:
                raise ValueError(
                    "Cannot auto-match feature dimension for NetCDF input: "
                    f"base_dim={base_dim}, anom_dim={anom_dim}, ckpt_input_dim={expected_input_dim}. "
                    "Use --append-annual-anomalies / --no-append-annual-anomalies explicitly."
                )
        feat = np.transpose(np.asarray(feat_da.values, dtype=np.float32), (2, 0, 1, 3))
    finally:
        ds.close()
    return feat, feat_names, [str(p) for p in paths]


def load_nc_variable_array(
    *,
    source_paths: list[str],
    var_name: str,
) -> np.ndarray:
    if not source_paths:
        raise ValueError("source_paths is empty; cannot load full-trend physics variable.")
    ds = xr.open_mfdataset(source_paths, combine="by_coords", engine="netcdf4")
    try:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": TIME_DIM})
        if "number" in ds.coords or "number" in ds:
            ds = ds.drop_vars("number", errors="ignore")
        if "pressure_level" in ds.dims:
            ds = ds.isel(pressure_level=0).drop_vars("pressure_level", errors="ignore")
        if "level" in ds.dims:
            ds = ds.isel(level=0).drop_vars("level", errors="ignore")
        if var_name not in ds.data_vars:
            raise KeyError(f"Required full-trend physics variable '{var_name}' not found in {source_paths}")
        da = ds[var_name]
        required_dims = {"latitude", "longitude", TIME_DIM}
        missing = required_dims - set(da.dims)
        if missing:
            raise ValueError(f"Variable '{var_name}' missing required dims: {sorted(missing)}")
        da = da.transpose("latitude", "longitude", TIME_DIM)
        arr = np.asarray(da.values, dtype=np.float32)
        return np.transpose(arr, (2, 0, 1))[..., None]
    finally:
        ds.close()


def load_lat_lon_from_nc(source_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(source_paths, combine="by_coords", engine="netcdf4")
    try:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": TIME_DIM})
        lat_name = "latitude" if "latitude" in ds.coords else "lat"
        lon_name = "longitude" if "longitude" in ds.coords else "lon"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            raise KeyError(f"Missing latitude/longitude coordinates in {source_paths}")
        return np.asarray(ds[lat_name].values, dtype=np.float32), np.asarray(ds[lon_name].values, dtype=np.float32)
    finally:
        ds.close()


def _compute_diffusion_aux_features_for_shap(
    *,
    aod_prev: torch.Tensor,
    u_prev: torch.Tensor,
    v_prev: torch.Tensor,
    physics_grid: SphericalGrid,
    contrast_window: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
    aod = aod_prev[..., 0]
    u = u_prev[..., 0]
    v = v_prev[..., 0]
    aod_adv = spherical_advection(aod, u, v, physics_grid) * 86400.0
    aod_adv = torch.nan_to_num(aod_adv, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(-1)

    half = int(contrast_window) // 2
    aod_4d = aod.unsqueeze(1)
    pad_lon = F.pad(aod_4d, (half, half, 0, 0), mode="circular")
    pad_full = F.pad(pad_lon, (0, 0, half, half), mode="replicate")
    neigh_mean = F.avg_pool2d(pad_full, kernel_size=int(contrast_window), stride=1)
    aod_con = torch.nan_to_num(aod_4d - neigh_mean, nan=0.0, posinf=0.0, neginf=0.0)
    return aod_adv, aod_con.squeeze(1).unsqueeze(-1)


def build_diffusion_task_model_from_ckpt(ckpt: dict, device: torch.device) -> torch.nn.Module | None:
    state = ckpt.get("diffusion_task_state_dict")
    if state is None:
        return None
    hidden = int(ckpt.get("diffusion_mlp_hidden", TRAIN_SETTINGS.get("diffusion_mlp_hidden", 16)))
    model = DiffusionTaskMLP(
        DiffusionTaskMLPConfig(
            input_dim=5,
            hidden_dims=(hidden, hidden),
            dropout=0.0,
        )
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def compute_online_physics_tendency_array(
    *,
    x: np.ndarray,
    feature_names: list[str],
    source_paths: list[str],
    ckpt: dict,
    device: torch.device,
    dt_days: float,
    advection_scheme: str,
    enable_diffusion: bool,
    diffusion_k_const: float,
    cos_floor: float,
    batch_days: int,
) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"Online full_trend physics expects [time, lat, lon, feature], got {x.shape}")
    aod_idx = int(find_feature_index(feature_names, "AOD", required=True))
    u_idx = int(find_feature_index(feature_names, "U10", required=True))
    v_idx = int(find_feature_index(feature_names, "V10", required=True))
    ws_idx = int(find_feature_index(feature_names, "WS", required=True))

    lat, lon = load_lat_lon_from_nc(source_paths)
    grid = SphericalGrid(
        lat_deg=torch.from_numpy(lat).to(device=device, dtype=torch.float32),
        lon_deg=torch.from_numpy(lon).to(device=device, dtype=torch.float32),
        cos_floor=float(cos_floor),
    )
    diffusion_task_model = build_diffusion_task_model_from_ckpt(ckpt, device)
    learn_diffusion = diffusion_task_model is not None and bool(ckpt.get("learn_diffusion_coeff", True))
    physics_cfg = PhysicsConfig(
        dt_days=float(dt_days),
        advection_scheme=str(advection_scheme),
        enable_diffusion=bool(enable_diffusion) or bool(learn_diffusion),
        diffusion_k_const=float(diffusion_k_const),
    )

    diffusion_k_min = float(ckpt.get("diffusion_k_min", TRAIN_SETTINGS.get("diffusion_k_min", 0.0)))
    diffusion_k_base = float(
        ckpt.get(
            "diffusion_k_base",
            TRAIN_SETTINGS.get("diffusion_k_base", PHYSICS_SETTINGS.get("diffusion_k_const", 1e-5)),
        )
    )
    diffusion_k_slope = float(ckpt.get("diffusion_k_slope", TRAIN_SETTINGS.get("diffusion_k_slope", 5e-6)))
    diffusion_k_cap = float(ckpt.get("diffusion_k_cap", TRAIN_SETTINGS.get("diffusion_k_cap", 5e-4)))

    out = np.empty(x.shape[:3] + (1,), dtype=np.float32)
    step = max(1, int(batch_days))
    for start in range(0, x.shape[0], step):
        stop = min(start + step, x.shape[0])
        chunk = np.nan_to_num(x[start:stop], nan=0.0, posinf=0.0, neginf=0.0)
        aod_prev = torch.from_numpy(chunk[..., aod_idx : aod_idx + 1]).to(device=device, dtype=torch.float32)
        u_prev = torch.from_numpy(chunk[..., u_idx : u_idx + 1]).to(device=device, dtype=torch.float32)
        v_prev = torch.from_numpy(chunk[..., v_idx : v_idx + 1]).to(device=device, dtype=torch.float32)
        ws_prev = torch.from_numpy(chunk[..., ws_idx : ws_idx + 1]).to(device=device, dtype=torch.float32)

        k_eff = None
        if diffusion_task_model is not None:
            aod_adv_feat, aod_con_feat = _compute_diffusion_aux_features_for_shap(
                aod_prev=aod_prev,
                u_prev=u_prev,
                v_prev=v_prev,
                physics_grid=grid,
            )
            diff_in = torch.cat([u_prev, v_prev, ws_prev, aod_adv_feat, aod_con_feat], dim=-1)
            with torch.no_grad():
                kappa_raw = diffusion_task_model(diff_in)
            if learn_diffusion:
                ws_mag = torch.abs(ws_prev)
                k_upper = float(diffusion_k_base) + float(diffusion_k_slope) * ws_mag
                k_upper = torch.clamp(
                    k_upper,
                    min=float(diffusion_k_min) + 1e-12,
                    max=max(float(diffusion_k_cap), float(diffusion_k_min) + 1e-12),
                )
                k_eff_full = float(diffusion_k_min) + (k_upper - float(diffusion_k_min)) * torch.sigmoid(kappa_raw)
                k_eff = torch.clamp(k_eff_full[..., 0], min=float(diffusion_k_min))

        with torch.no_grad():
            tend = physics_tendency(
                aod_prev[..., 0],
                u_prev[..., 0],
                v_prev[..., 0],
                grid,
                cfg=physics_cfg,
                k_eff=k_eff,
            )
            trend = tend["tendency_total"].detach().cpu().numpy().astype(np.float32)
        out[start:stop, ..., 0] = trend
        logging.info("Computed online full_trend physics tendency for time slices %d:%d", start, stop)
    return out


def load_feature_array_from_npz(npz_path: Path) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    data = np.load(npz_path)
    blocks: list[np.ndarray] = []
    for key in ("x_train", "x_val", "x_test"):
        if key in data:
            blocks.append(np.asarray(data[key], dtype=np.float32))
    if not blocks:
        raise KeyError("No x_train/x_val/x_test found in NPZ.")
    x = blocks[0]
    for b in blocks[1:]:
        if b.ndim == x.ndim and b.shape[1:] == x.shape[1:]:
            x = np.concatenate([x, b], axis=0)
    return x


def feature_matrix(x: np.ndarray, *, sample_step: int) -> np.ndarray:
    if x.ndim == 5:
        step = max(0, min(int(sample_step), int(x.shape[1]) - 1))
        x = x[:, step, ...]
    if x.ndim == 4:
        x2d = x.reshape(-1, x.shape[-1])
    elif x.ndim == 3:
        x2d = x.reshape(-1, x.shape[-1])
    elif x.ndim == 2:
        x2d = x
    else:
        raise ValueError(f"Unsupported feature array shape: {x.shape}")
    return np.nan_to_num(np.asarray(x2d, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def find_feature_index(feature_names: list[str], key: str, *, required: bool = True) -> int | None:
    cands = FEATURE_CANDIDATES.get(key.upper())
    if cands is None:
        if required:
            raise KeyError(f"Unknown feature key: {key}")
        return None
    for i, name in enumerate(feature_names):
        if name in cands or str(name).upper() == key.upper():
            return i
    if required:
        raise KeyError(f"Required feature '{key}' not found in feature_names={feature_names}")
    return None


def build_model(ckpt: dict, feature_names: list[str], device: torch.device) -> tuple[torch.nn.Module, str]:
    state = ckpt["model_state_dict"]
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", [128, 128]))
    dropout = float(ckpt.get("dropout", 0.1))
    input_dim = int(ckpt.get("input_dim", len(feature_names)))
    if any(k.startswith("rad_trunk.") for k in state.keys()):
        cfg = ProcessDecomposedMLPConfig(
            feature_dim=input_dim,
            aod_index=int(find_feature_index(feature_names, "AOD", required=True)),
            u10_index=int(find_feature_index(feature_names, "U10", required=True)),
            v10_index=int(find_feature_index(feature_names, "V10", required=True)),
            wind_speed_index=int(find_feature_index(feature_names, "WS", required=True)),
            ceres_index=int(find_feature_index(feature_names, "CERES_ALL", required=True)),
            sza_index=find_feature_index(feature_names, "SZA", required=False),
            t2m_index=int(find_feature_index(feature_names, "T2M", required=True)),
            blh_index=int(find_feature_index(feature_names, "BLH", required=True)),
            tp_index=int(find_feature_index(feature_names, "TP", required=True)),
            clw_index=int(find_feature_index(feature_names, "CLW", required=True)),
            rh_index=int(find_feature_index(feature_names, "R", required=True)),
            sp_index=int(find_feature_index(feature_names, "SP", required=True)),
            ceres_anom_index=find_feature_index(feature_names, "CERES_ANOM", required=False),
            t2m_anom_index=find_feature_index(feature_names, "T2M_ANOM", required=False),
            rh_anom_index=find_feature_index(feature_names, "R_ANOM", required=False),
            hidden_dims=hidden_dims,
            dropout=dropout,
            rad_init_bias=float(ckpt.get("rad_init_bias", -1.0)),
            dry_init_bias=float(ckpt.get("dry_init_bias", -3.0)),
            wet_init_bias=float(ckpt.get("wet_init_bias", -3.0)),
        )
        model: torch.nn.Module = ProcessDecomposedMLP(cfg)
        model_type = "process_decomposed"
    else:
        model = GridPointMLP(MLPConfig(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, output_dim=1))
        model_type = "gridpoint_mlp"
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model, model_type


class WrappedD2Model(torch.nn.Module):
    def __init__(
        self,
        base_model: torch.nn.Module,
        target: str,
        model_type: str,
        *,
        model_input_dim: int | None = None,
        physics_feature_index: int | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.target = target
        self.model_type = model_type
        self.model_input_dim = model_input_dim
        self.physics_feature_index = physics_feature_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor
        model_x = x
        if self.model_input_dim is not None:
            model_x = x[..., : int(self.model_input_dim)]
        if self.target == "model_output":
            y = self.base_model(model_x)
        else:
            if self.model_type != "process_decomposed":
                raise ValueError(f"target={self.target} requires process_decomposed model.")
            comps = self.base_model.forward_components(model_x)
            if self.target in ("aod_src", "trend_rad"):
                y = comps["trend_rad"]
            elif self.target in ("aod_dep", "trend_dep"):
                y = -(comps["k_dep"] * comps["aod_ref"])
            elif self.target == "trend_total":
                y = comps["trend_rad"] - (comps["k_dep"] * comps["aod_ref"])
            elif self.target == FULL_TREND_TARGET:
                if self.physics_feature_index is None:
                    raise ValueError("full_trend requires physics_feature_index.")
                physics_trend = x[..., int(self.physics_feature_index) : int(self.physics_feature_index) + 1]
                y = physics_trend + comps["trend_rad"] - (comps["k_dep"] * comps["aod_ref"])
            elif self.target == "t2m_next":
                y = comps["t2m_next"]
            elif self.target == "ceres_next":
                y = comps["ceres_next"]
            elif self.target == "rh_next":
                y = comps["rh_next"]
            elif self.target == "k_dep":
                y = comps["k_dep"]
            elif self.target == "dep_term":
                y = comps["k_dep"] * comps["aod_ref"]
            else:
                raise ValueError(f"Unsupported target: {self.target}")
        return y.reshape(y.shape[0], -1).mean(dim=1, keepdim=True)


def _stack_shap_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    if not chunks:
        raise ValueError("No SHAP chunks were produced.")
    return np.concatenate(chunks, axis=0)


def _to_shap_array(shap_values_raw: object) -> np.ndarray:
    shap_values = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw
    arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    need_full_trend = bool(args.include_full_trend) or str(args.target) == FULL_TREND_TARGET

    if args.data_npz is not None:
        if need_full_trend:
            raise ValueError("full_trend SHAP requires NetCDF input so the physics tendency variable can be loaded.")
        x = load_feature_array_from_npz(args.data_npz)
        source_paths = [str(args.data_npz)]
        feature_names = list(ckpt.get("feature_names") or ckpt.get("feature_keys") or args.feature_keys)
    else:
        expected_input_dim = int(ckpt.get("input_dim")) if ckpt.get("input_dim") is not None else None
        x, feature_names, source_paths = load_feature_array_from_nc(
            data_nc=args.data_nc,
            data_root=args.data_root,
            years=args.years,
            feature_keys=list(args.feature_keys),
            require_all_features=bool(args.require_all_features),
            append_annual_anomalies=args.append_annual_anomalies,
            expected_input_dim=expected_input_dim,
        )

    x2d = feature_matrix(x, sample_step=int(args.sample_step))
    if x2d.shape[1] != int(ckpt.get("input_dim", x2d.shape[1])):
        raise ValueError(f"Feature dim mismatch between data and ckpt: {x2d.shape[1]} vs {ckpt.get('input_dim')}")

    raw_rows = int(x2d.shape[0])
    phys2d: np.ndarray | None = None
    full_trend_physics_feature_name = str(args.full_trend_physics_feature_name or args.full_trend_physics_var)
    if need_full_trend:
        phys_arr: np.ndarray | None = None
        if str(args.full_trend_physics_source) in {"auto", "var"}:
            try:
                phys_arr = load_nc_variable_array(
                    source_paths=source_paths,
                    var_name=str(args.full_trend_physics_var),
                )
                logging.info(
                    "Loaded full_trend physics tendency from NetCDF variable '%s'.",
                    str(args.full_trend_physics_var),
                )
            except Exception as exc:
                if str(args.full_trend_physics_source) == "var":
                    raise
                logging.warning(
                    "Could not load full_trend physics variable '%s' (%s). Falling back to online physics.",
                    str(args.full_trend_physics_var),
                    exc,
                )
        if phys_arr is None:
            if args.full_trend_physics_feature_name is None:
                full_trend_physics_feature_name = "online_physics_tendency"
            phys_arr = compute_online_physics_tendency_array(
                x=x,
                feature_names=feature_names,
                source_paths=source_paths,
                ckpt=ckpt,
                device=device,
                dt_days=float(args.dt_days),
                advection_scheme=str(args.physics_advection_scheme),
                enable_diffusion=bool(args.physics_enable_diffusion),
                diffusion_k_const=float(args.physics_diffusion_k_const),
                cos_floor=float(args.physics_cos_floor),
                batch_days=int(args.full_trend_online_batch_days),
            )
        phys2d = feature_matrix(phys_arr, sample_step=int(args.sample_step))
        if phys2d.shape[0] != raw_rows:
            raise ValueError(
                f"Physics tendency row count mismatch: features={raw_rows}, "
                f"{args.full_trend_physics_var}={phys2d.shape[0]}"
            )
        if phys2d.shape[1] != 1:
            raise ValueError(f"Expected one physics tendency column, got shape={phys2d.shape}")

    if args.max_rows > 0 and x2d.shape[0] > args.max_rows:
        keep = np.random.choice(x2d.shape[0], size=int(args.max_rows), replace=False)
        x2d = x2d[keep]
        if phys2d is not None:
            phys2d = phys2d[keep]

    if x2d.shape[0] < 2:
        raise ValueError(f"Not enough rows for SHAP after filtering: {x2d.shape[0]}")

    n_bg = min(int(args.background_size), x2d.shape[0])
    remain = max(1, x2d.shape[0] - n_bg)
    if int(args.explain_size) <= 0:
        n_exp = remain
    else:
        n_exp = min(int(args.explain_size), remain)
    perm = np.random.permutation(x2d.shape[0])
    bg = x2d[perm[:n_bg]]
    ex = x2d[perm[n_bg : n_bg + n_exp]]
    bg_full = ex_full = None
    full_feature_names = None
    full_physics_feature_index = None
    if phys2d is not None:
        x2d_full = np.concatenate([x2d, phys2d], axis=1)
        bg_full = x2d_full[perm[:n_bg]]
        ex_full = x2d_full[perm[n_bg : n_bg + n_exp]]
        full_feature_names = list(feature_names) + [full_trend_physics_feature_name]
        full_physics_feature_index = len(full_feature_names) - 1

    model, model_type = build_model(ckpt, feature_names, device)
    if bool(args.all_contrib) and bool(args.all_tasks):
        raise ValueError("--all-contrib and --all-tasks cannot be enabled together.")
    if bool(args.all_contrib):
        if model_type != "process_decomposed":
            raise ValueError("--all-contrib requires process_decomposed checkpoint.")
        target_list = list(D2_PROCESS_CONTRIB_TARGETS)
        if bool(args.include_full_trend):
            target_list.append(FULL_TREND_TARGET)
    elif bool(args.all_tasks):
        if model_type != "process_decomposed":
            raise ValueError("--all-tasks requires process_decomposed checkpoint.")
        target_list = list(D2_MULTI_TASK_TARGETS)
    else:
        target_list = [str(args.target)]
    if model_type != "process_decomposed":
        for t in target_list:
            if t != "model_output":
                raise ValueError(f"target={t} is only valid for process_decomposed checkpoints.")
    if FULL_TREND_TARGET in target_list and phys2d is None:
        raise ValueError("full_trend target requires --include-full-trend or --target full_trend with NetCDF inputs.")

    try:
        import shap
    except Exception as exc:
        raise RuntimeError("shap is required. Please install with: pip install shap") from exc

    for target in target_list:
        if str(target) == FULL_TREND_TARGET:
            if bg_full is None or ex_full is None or full_feature_names is None or full_physics_feature_index is None:
                raise ValueError("full_trend target was requested but full-trend input arrays were not built.")
            target_bg = bg_full
            target_ex = ex_full
            target_feature_names = full_feature_names
            wrapped = WrappedD2Model(
                model,
                target=str(target),
                model_type=model_type,
                model_input_dim=int(ckpt.get("input_dim", len(feature_names))),
                physics_feature_index=int(full_physics_feature_index),
            ).to(device)
        else:
            target_bg = bg
            target_ex = ex
            target_feature_names = feature_names
            wrapped = WrappedD2Model(model, target=str(target), model_type=model_type).to(device)
        wrapped.eval()
        logging.info(
            "Explainer=%s | model=%s | target=%s | rows_raw=%d | rows_used=%d | bg=%d | explain=%d | chunk=%d",
            args.explainer,
            model_type,
            target,
            raw_rows,
            int(x2d.shape[0]),
            n_bg,
            n_exp,
            int(args.explain_batch_size),
        )

        if args.explainer == "gradient":
            bg_t = torch.from_numpy(target_bg).to(device=device, dtype=torch.float32)
            explainer = shap.GradientExplainer(wrapped, bg_t)
            chunk = max(1, int(args.explain_batch_size))
            shap_chunks: list[np.ndarray] = []
            for i in range(0, target_ex.shape[0], chunk):
                ex_t = torch.from_numpy(target_ex[i : i + chunk]).to(device=device, dtype=torch.float32)
                shap_raw_chunk = explainer.shap_values(ex_t, nsamples=int(args.nsamples))
                shap_chunks.append(_to_shap_array(shap_raw_chunk))
            shap_values = _stack_shap_chunks(shap_chunks)
        else:
            def _predict(a: np.ndarray) -> np.ndarray:
                out = []
                for i in range(0, a.shape[0], int(args.batch_size)):
                    b = torch.from_numpy(np.asarray(a[i : i + int(args.batch_size)], dtype=np.float32)).to(device)
                    with torch.no_grad():
                        y = wrapped(b).detach().cpu().numpy().reshape(-1)
                    out.append(y)
                return np.concatenate(out, axis=0)

            explainer = shap.KernelExplainer(_predict, target_bg)
            chunk = max(1, int(args.explain_batch_size))
            shap_chunks = []
            for i in range(0, target_ex.shape[0], chunk):
                shap_raw_chunk = explainer.shap_values(target_ex[i : i + chunk], nsamples=int(args.nsamples))
                shap_chunks.append(_to_shap_array(shap_raw_chunk))
            shap_values = _stack_shap_chunks(shap_chunks)

        if shap_values.shape[1] != len(target_feature_names):
            raise ValueError(f"SHAP dim mismatch: shap={shap_values.shape}, feature_names={len(target_feature_names)}")

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        mean_signed = np.mean(shap_values, axis=0)
        order = np.argsort(-mean_abs)

        out_dir = args.out_dir / f"{args.ckpt.stem}_{target}"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "shap_values.npy", shap_values.astype(np.float32))
        np.save(out_dir / "explain_samples.npy", target_ex.astype(np.float32))

        lines = ["feature,mean_abs_shap,mean_signed_shap,rank"]
        for rank, idx in enumerate(order, start=1):
            lines.append(f"{target_feature_names[idx]},{mean_abs[idx]:.10e},{mean_signed[idx]:.10e},{rank}")
        (out_dir / "feature_importance.csv").write_text("\n".join(lines), encoding="utf-8")

        meta = {
            "framework": "D2",
            "ckpt": str(args.ckpt),
            "model_type": model_type,
            "prediction_mode": str(ckpt.get("prediction_mode", "unknown")),
            "target": target,
            "all_contrib": bool(args.all_contrib),
            "all_tasks": bool(args.all_tasks),
            "include_full_trend": bool(args.include_full_trend),
            "feature_names": target_feature_names,
            "rows_raw": raw_rows,
            "rows_used": int(x2d.shape[0]),
            "max_rows": int(args.max_rows),
            "background_size": int(n_bg),
            "explain_size": int(n_exp),
            "explain_batch_size": int(args.explain_batch_size),
            "explainer": args.explainer,
            "nsamples": int(args.nsamples),
            "data_sources": source_paths,
            "sample_step": int(args.sample_step),
        }
        if str(target) == FULL_TREND_TARGET:
            meta.update(
                {
                    "full_trend_definition": (
                        f"full_trend = {args.full_trend_physics_var} + trend_rad - k_dep * aod_ref"
                    ),
                    "full_trend_physics_var": str(args.full_trend_physics_var),
                    "full_trend_physics_source": str(args.full_trend_physics_source),
                    "full_trend_physics_feature_name": full_trend_physics_feature_name,
                    "full_trend_physics_feature_index": int(full_physics_feature_index),
                    "full_trend_note": (
                        "The physics tendency is loaded as a precomputed NetCDF variable and appended as one "
                        "extra SHAP feature. It is not decomposed back into the U/V/AOD spatial-operator inputs."
                    ),
                }
            )
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

        try:
            import matplotlib.pyplot as plt

            topk = min(int(args.topk), len(target_feature_names))
            top_idx = order[:topk][::-1]
            plt.figure(figsize=(10, 0.45 * topk + 2))
            plt.barh([target_feature_names[i] for i in top_idx], [mean_abs[i] for i in top_idx])
            plt.xlabel("mean(|SHAP|)")
            plt.title(f"D2 SHAP ({model_type}, target={target})")
            plt.tight_layout()
            plt.savefig(out_dir / "top_features_bar.png", dpi=180)
            plt.close()

            shap.summary_plot(shap_values, target_ex, feature_names=target_feature_names, show=False)
            plt.tight_layout()
            plt.savefig(out_dir / "shap_beeswarm.png", dpi=180)
            plt.close()
        except Exception as exc:
            logging.warning("Plot generation skipped for target=%s: %s", target, exc)

        logging.info("Saved SHAP outputs to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
