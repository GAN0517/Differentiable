from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch import nn
from torch.utils.data import DataLoader

# Allow running as both:
# 1) python Difference/train/train_mlp_baseline.py
# 2) python -m Difference.train.train_mlp_baseline
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from Difference.dl import (
    FeatureTargetDataset,
    GridPointMLP,
    MLPConfig,
)
from Difference.settings import GLOBAL_SETTINGS, PHYSICS_SETTINGS, TRAIN_SETTINGS
from Difference.train.losses import (
    FORCE_ZERO_DL_OUTPUT,
    LossConfig,
    compute_autoregressive_trend_loss,
    compute_coupled_loss,
)

TIME_DIM = "time"

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
    "SZA": ("ceres_sza", "sza", "solar_zenith_angle"),
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


def _resolve_run_seeds(args: argparse.Namespace) -> list[int]:
    if args.seed is not None:
        return [int(args.seed)]
    seeds = [int(s) for s in args.seed_list]
    if args.num_models <= 0:
        raise ValueError("--num-models must be > 0.")
    if len(seeds) < args.num_models:
        raise ValueError(
            f"--seed-list has {len(seeds)} values, but --num-models={args.num_models}. "
            "Please provide at least num_models seeds."
        )
    return seeds[: args.num_models]


def _path_with_seed_suffix(path: Path, seed: int) -> Path:
    return path.with_name(f"{path.stem}_seed{seed}{path.suffix}")


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D5 pure-DL single-MLP trend baseline (physics disabled).")
    parser.add_argument("--data-npz", type=Path, help="NPZ containing x_train/y_train/x_val/y_val.")
    parser.add_argument("--data-nc", type=Path, help="Single NetCDF file, e.g. data_normalize.nc.")
    parser.add_argument("--data-root", type=Path, help="Root containing data_normalize.nc or <year>/data_normalize.nc.")
    parser.add_argument("--years", type=int, nargs="+", default=list(TRAIN_SETTINGS.get("years", GLOBAL_SETTINGS["years"])), help="Years for --data-root mode.")

    parser.add_argument("--target-var", type=str, default=TRAIN_SETTINGS["target_var"])
    parser.add_argument("--physics-var", type=str, help="Physics baseline variable used in delta mode.")
    parser.add_argument("--prediction-mode", choices=("direct", "trend_sum"), default=TRAIN_SETTINGS["prediction_mode"])
    parser.add_argument("--horizon", type=int, default=TRAIN_SETTINGS["horizon"])
    parser.add_argument("--rollout-steps", type=int, default=TRAIN_SETTINGS["rollout_steps"], help="Autoregressive rollout steps (trend_sum mode).")
    parser.add_argument("--dt-days", type=float, default=TRAIN_SETTINGS["dt_days"], help="Time step in days used for iterative state update.")
    parser.add_argument("--rollout-decay", type=float, default=TRAIN_SETTINGS["rollout_decay"], help="Time-decay gamma for autoregressive loss weights.")
    parser.add_argument(
        "--online-physics-coupling",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAIN_SETTINGS.get("online_physics_coupling", True)),
        help="If enabled, recompute physics tendency online each rollout step using previous predicted state.",
    )
    parser.add_argument(
        "--iterative-aux-forcing",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAIN_SETTINGS.get("iterative_aux_forcing", True)),
        help=(
            "If enabled, collaborative forcing states (U/V/T2M/CERES/RH) are fed back autoregressively across rollout steps. "
            "If disabled, they use baseline forcing from the current step only."
        ),
    )
    parser.add_argument(
        "--predict-uv10",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="[Deprecated] UV next-state prediction task has been removed and this flag is ignored.",
    )
    parser.add_argument(
        "--physics-advection-scheme",
        choices=("semi_lagrangian", "central"),
        default=str(PHYSICS_SETTINGS.get("scheme", "semi_lagrangian")),
        help="Advection scheme for online physics coupling.",
    )
    parser.add_argument(
        "--physics-enable-diffusion",
        action=argparse.BooleanOptionalAction,
        default=bool(PHYSICS_SETTINGS.get("enable_diffusion", False)),
        help="Enable diffusion term in online physics coupling.",
    )
    parser.add_argument(
        "--physics-diffusion-k-const",
        type=float,
        default=float(PHYSICS_SETTINGS.get("diffusion_k_const", 0.0)),
        help="Constant diffusion coefficient K for online physics coupling.",
    )
    parser.add_argument(
        "--physics-cos-floor",
        type=float,
        default=float(PHYSICS_SETTINGS.get("cos_floor", 0.05)),
        help="Cos(latitude) floor for spherical operators in online physics coupling.",
    )
    parser.add_argument(
        "--learn-diffusion-coeff",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAIN_SETTINGS.get("learn_diffusion_coeff", False)),
        help="Learn per-grid diffusion coefficient from wind-related features in online physics coupling.",
    )
    parser.add_argument(
        "--diffusion-mlp-hidden",
        type=int,
        default=int(TRAIN_SETTINGS.get("diffusion_mlp_hidden", 16)),
        help="Hidden width of diffusion coefficient MLP.",
    )
    parser.add_argument(
        "--diffusion-mlp-lr",
        type=float,
        default=float(TRAIN_SETTINGS.get("diffusion_mlp_lr", TRAIN_SETTINGS.get("lr_src", TRAIN_SETTINGS["lr"]))),
        help="Learning rate of diffusion coefficient MLP.",
    )
    parser.add_argument(
        "--diffusion-k-min",
        type=float,
        default=float(TRAIN_SETTINGS.get("diffusion_k_min", 0.0)),
        help="Lower bound of learned diffusion coefficient.",
    )
    parser.add_argument(
        "--diffusion-k-base",
        type=float,
        default=float(TRAIN_SETTINGS.get("diffusion_k_base", PHYSICS_SETTINGS.get("diffusion_k_const", 1e-5))),
        help="Base upper-bound term for learned diffusion coefficient.",
    )
    parser.add_argument(
        "--diffusion-k-slope",
        type=float,
        default=float(TRAIN_SETTINGS.get("diffusion_k_slope", 5e-6)),
        help="Wind-dependent slope for learned diffusion coefficient upper bound.",
    )
    parser.add_argument(
        "--diffusion-k-cap",
        type=float,
        default=float(TRAIN_SETTINGS.get("diffusion_k_cap", 5e-4)),
        help="Global cap for learned diffusion coefficient.",
    )
    parser.add_argument("--dev-frac", type=float, default=float(TRAIN_SETTINGS.get("dev_frac", 0.8)), help="Front portion used for train+val development split.")
    parser.add_argument("--train-frac", type=float, default=TRAIN_SETTINGS["train_frac"], help="Train fraction inside development split.")
    parser.add_argument("--feature-keys", type=str, nargs="+", default=DEFAULT_FEATURE_KEYS)
    parser.add_argument(
        "--require-all-features",
        action=argparse.BooleanOptionalAction,
        default=bool(TRAIN_SETTINGS.get("require_all_features", False)),
        help="If enabled, missing feature keys will raise an error. Otherwise missing ones are skipped with warning.",
    )

    parser.add_argument("--epochs", type=int, default=TRAIN_SETTINGS["epochs"])
    parser.add_argument("--eval-interval", type=int, default=TRAIN_SETTINGS["eval_interval"])
    parser.add_argument("--patience", type=int, default=TRAIN_SETTINGS["patience"])
    parser.add_argument("--min-delta", type=float, default=TRAIN_SETTINGS["min_delta"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_SETTINGS["batch_size"])
    parser.add_argument("--max-points-per-forward", type=int, default=TRAIN_SETTINGS["max_points_per_forward"])
    parser.add_argument("--lr", type=float, default=TRAIN_SETTINGS["lr"])
    parser.add_argument("--lr-src", type=float, default=float(TRAIN_SETTINGS.get("lr_src", TRAIN_SETTINGS["lr"])))
    parser.add_argument("--lr-wet", type=float, default=float(TRAIN_SETTINGS.get("lr_wet", TRAIN_SETTINGS["lr"])))
    parser.add_argument(
        "--lr-dry",
        type=float,
        default=float(TRAIN_SETTINGS.get("lr_dry", TRAIN_SETTINGS["lr"] * 0.1)),
    )
    parser.add_argument("--dropout", type=float, default=TRAIN_SETTINGS["dropout"])
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=list(TRAIN_SETTINGS["hidden_dims"]))
    parser.add_argument("--mse-weight", type=float, default=TRAIN_SETTINGS["mse_weight"])
    parser.add_argument("--nonneg-weight", type=float, default=TRAIN_SETTINGS["nonneg_weight"])
    parser.add_argument(
        "--sparsity-weight",
        type=float,
        default=float(TRAIN_SETTINGS.get("sparsity_weight", 1e-4)),
        help="L1 sparsity penalty weight for process branch outputs in trend_sum mode.",
    )
    parser.add_argument(
        "--aod-task-weight",
        type=float,
        default=float(TRAIN_SETTINGS.get("aod_task_weight", LossConfig().aod_task_weight)),
        help="Task weight for AOD rollout supervision in trend_sum mode.",
    )
    parser.add_argument(
        "--t2m-task-weight",
        type=float,
        default=float(TRAIN_SETTINGS.get("t2m_task_weight", LossConfig().t2m_task_weight)),
        help="Task weight for auxiliary T2M next-state supervision in trend_sum mode.",
    )
    parser.add_argument(
        "--ceres-task-weight",
        type=float,
        default=float(TRAIN_SETTINGS.get("ceres_task_weight", LossConfig().ceres_task_weight)),
        help="Task weight for auxiliary CERES next-state supervision in trend_sum mode.",
    )
    parser.add_argument(
        "--rh-task-weight",
        type=float,
        default=float(TRAIN_SETTINGS.get("rh_task_weight", LossConfig().rh_task_weight)),
        help="Task weight for auxiliary RH next-state supervision in trend_sum mode.",
    )
    parser.add_argument("--rad-init-bias", type=float, default=float(TRAIN_SETTINGS.get("rad_init_bias", -1.0)))
    parser.add_argument("--dry-init-bias", type=float, default=float(TRAIN_SETTINGS.get("dry_init_bias", -3.0)))
    parser.add_argument("--wet-init-bias", type=float, default=float(TRAIN_SETTINGS.get("wet_init_bias", -3.0)))
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=TRAIN_SETTINGS["device"] if (TRAIN_SETTINGS["device"] == "cpu" or torch.cuda.is_available()) else "cpu",
    )
    parser.add_argument("--cuda-device-id", type=int, default=int(TRAIN_SETTINGS["cuda_device_id"]))
    parser.add_argument("--seed", type=int, default=None, help="Single-seed override for one training run.")
    parser.add_argument(
        "--num-models",
        type=int,
        default=int(TRAIN_SETTINGS.get("num_models", 1)),
        help="How many models to train in one execution (one seed per model).",
    )
    parser.add_argument(
        "--seed-list",
        type=int,
        nargs="+",
        default=list(TRAIN_SETTINGS.get("seed_list", [42])),
        help="Seed list used when --seed is not provided; first num-models seeds are used.",
    )
    parser.add_argument("--single-seed-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, default=Path(TRAIN_SETTINGS["out_ckpt"]))
    parser.add_argument(
        "--eval-nc-out",
        type=Path,
        help="Optional NetCDF path to save final test predictions/targets.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    if args.data_npz is None and TRAIN_SETTINGS.get("train_data_npz") is not None:
        args.data_npz = Path(TRAIN_SETTINGS["train_data_npz"])
    if args.data_nc is None and TRAIN_SETTINGS.get("train_data_nc") is not None:
        args.data_nc = Path(TRAIN_SETTINGS["train_data_nc"])
    if args.data_root is None:
        if TRAIN_SETTINGS.get("train_data_root") is not None:
            args.data_root = Path(TRAIN_SETTINGS["train_data_root"])
        else:
            args.data_root = Path(GLOBAL_SETTINGS["data_root"])
    if float(args.physics_cos_floor) <= 0.0:
        raise ValueError(f"--physics-cos-floor must be > 0, got {args.physics_cos_floor}")
    if float(args.physics_diffusion_k_const) < 0.0:
        raise ValueError(f"--physics-diffusion-k-const must be >= 0, got {args.physics_diffusion_k_const}")
    if int(args.diffusion_mlp_hidden) <= 0:
        raise ValueError(f"--diffusion-mlp-hidden must be > 0, got {args.diffusion_mlp_hidden}")
    if float(args.diffusion_mlp_lr) <= 0.0:
        raise ValueError(f"--diffusion-mlp-lr must be > 0, got {args.diffusion_mlp_lr}")
    if float(args.diffusion_k_min) < 0.0:
        raise ValueError(f"--diffusion-k-min must be >= 0, got {args.diffusion_k_min}")
    if float(args.diffusion_k_base) < float(args.diffusion_k_min):
        raise ValueError(
            f"--diffusion-k-base must be >= diffusion_k_min ({args.diffusion_k_min}), got {args.diffusion_k_base}"
        )
    if float(args.diffusion_k_slope) < 0.0:
        raise ValueError(f"--diffusion-k-slope must be >= 0, got {args.diffusion_k_slope}")
    if float(args.diffusion_k_cap) <= float(args.diffusion_k_min):
        raise ValueError(
            f"--diffusion-k-cap must be > diffusion_k_min ({args.diffusion_k_min}), got {args.diffusion_k_cap}"
        )
    if int(args.num_models) <= 0:
        raise ValueError("--num-models must be > 0.")
    if args.seed is None and (not args.seed_list):
        raise ValueError("seed_list is empty. Set TRAIN_SETTINGS['seed_list'] or pass --seed.")
    # D5 hard policy: pure DL only, no physics process participation.
    if args.prediction_mode == "delta":
        raise ValueError("D5 is pure-DL. --prediction-mode=delta is disabled.")
    if bool(args.online_physics_coupling):
        logging.warning("D5 is pure-DL. Overriding --online-physics-coupling to false.")
    args.online_physics_coupling = False
    if args.physics_var is not None:
        logging.warning("D5 is pure-DL. Ignoring --physics-var=%s.", str(args.physics_var))
    args.physics_var = None
    if bool(args.learn_diffusion_coeff):
        logging.warning("D5 is pure-DL. Overriding --learn-diffusion-coeff to false.")
    args.learn_diffusion_coeff = False
    args.physics_enable_diffusion = False
    args.physics_diffusion_k_const = 0.0
    return args


def resolve_data_paths(base: Path, years: list[int] | None, filename: str) -> list[Path]:
    if not years:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return [path]
    paths: list[Path] = []
    for year in years:
        year_i = int(year)
        candidates = [
            base / str(year_i) / filename,
            base / f"D{year_i}" / filename,
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                "Required file not found for year %d. Tried: %s"
                % (year_i, ", ".join(str(p) for p in candidates))
            )
        paths.append(found)
    return paths


def load_combined_normalized_dataset(paths: list[Path]) -> xr.Dataset:
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords", engine="netcdf4")
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": TIME_DIM})
    if "number" in ds.coords or "number" in ds:
        ds = ds.drop_vars("number", errors="ignore")
    if "pressure_level" in ds.dims:
        ds = ds.isel(pressure_level=0).drop_vars("pressure_level")
    if "level" in ds.dims:
        ds = ds.isel(level=0).drop_vars("level")
    ds = ds.sortby(["latitude", "longitude", TIME_DIM])
    return ds


def resolve_feature_names(ds: xr.Dataset, feature_keys: list[str], require_all: bool) -> list[str]:
    names: list[str] = []
    missing: list[str] = []
    for key in feature_keys:
        key_up = key.upper()
        candidates = FEATURE_CANDIDATES.get(key_up)
        if candidates is None:
            missing.append(f"{key} (unknown key)")
            continue
        chosen = next((name for name in candidates if name in ds.data_vars), None)
        if chosen is None:
            missing.append(f"{key} -> {candidates}")
        else:
            names.append(chosen)
    if missing and require_all:
        raise KeyError("Missing required features: " + "; ".join(missing))
    if missing and not require_all:
        logging.warning("Skipping missing features: %s", "; ".join(missing))
    return names


def find_feature_index(
    used_features: list[str],
    key: str,
    *,
    required: bool = True,
) -> int | None:
    candidates = FEATURE_CANDIDATES.get(key.upper())
    if candidates is None:
        if required:
            raise KeyError(f"Unknown feature key: {key}")
        return None
    for idx, name in enumerate(used_features):
        if name in candidates:
            return idx
    if required:
        raise KeyError(f"Required feature key '{key}' not found in resolved features: {used_features}")
    return None


def _resolve_stats_paths_from_sources(source_paths: list[str]) -> list[Path]:
    stats_paths: list[Path] = []
    seen: set[Path] = set()
    for src in source_paths:
        candidate = Path(src).parent / "data_stats.nc"
        if candidate in seen:
            continue
        seen.add(candidate)
        stats_paths.append(candidate)
    return stats_paths


def resolve_target_denorm_params(
    *,
    target_var: str,
    stats_paths: list[Path],
) -> tuple[float | None, float | None, str]:
    """
    Resolve inverse-normalization params:
      target_physical = target_normalized * target_range + target_min
    Returns (target_min, target_range, source_msg).
    """
    min_name = f"{target_var}_min"
    range_name = f"{target_var}_range"
    for stats_path in stats_paths:
        if not stats_path.exists():
            continue
        try:
            with xr.open_dataset(stats_path) as stats_ds:
                if min_name not in stats_ds.data_vars or range_name not in stats_ds.data_vars:
                    continue
                target_min = float(np.nanmean(np.asarray(stats_ds[min_name].values)))
                target_range = float(np.nanmean(np.asarray(stats_ds[range_name].values)))
        except Exception as exc:
            logging.warning("Failed to read target denorm stats from %s: %s", stats_path, exc)
            continue

        if not np.isfinite(target_min) or not np.isfinite(target_range) or target_range <= 1e-12:
            continue
        return target_min, target_range, str(stats_path)

    return None, None, "not_found"



def resolve_var_denorm_params(
    *,
    var_name: str,
    stats_paths: list[Path],
) -> tuple[float | None, float | None, str]:
    """
    Generic inverse-normalization params:
      var_physical = var_normalized * var_range + var_min
    Returns (var_min, var_range, source_msg).
    """
    return resolve_target_denorm_params(target_var=var_name, stats_paths=stats_paths)


def _split_keep_grid_time(
    feature_4d: np.ndarray,
    target_3d: np.ndarray,
    train_frac: float,
    dev_frac: float,
    physics_3d: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    lat_n, lon_n, t_eff, feat_n = feature_4d.shape
    split_dev = int(math.floor(t_eff * dev_frac))
    if split_dev <= 1 or split_dev >= t_eff:
        raise ValueError(f"Invalid dev split: split_dev={split_dev}, t_eff={t_eff}, dev_frac={dev_frac}")
    split_train = int(math.floor(split_dev * train_frac))
    if split_train <= 0 or split_train >= split_dev:
        raise ValueError(
            f"Invalid train split inside dev: split_train={split_train}, split_dev={split_dev}, train_frac={train_frac}"
        )

    # Keep spatial grid intact: [T, Lat, Lon, ...]
    x_train = np.transpose(feature_4d[:, :, :split_train, :], (2, 0, 1, 3)).astype(np.float32)
    y_train = np.transpose(target_3d[:, :, :split_train], (2, 0, 1))[..., None].astype(np.float32)
    x_val = np.transpose(feature_4d[:, :, split_train:split_dev, :], (2, 0, 1, 3)).astype(np.float32)
    y_val = np.transpose(target_3d[:, :, split_train:split_dev], (2, 0, 1))[..., None].astype(np.float32)
    x_test = np.transpose(feature_4d[:, :, split_dev:, :], (2, 0, 1, 3)).astype(np.float32)
    y_test = np.transpose(target_3d[:, :, split_dev:], (2, 0, 1))[..., None].astype(np.float32)

    p_train = p_val = p_test = None
    if physics_3d is not None:
        p_train = np.transpose(physics_3d[:, :, :split_train], (2, 0, 1))[..., None].astype(np.float32)
        p_val = np.transpose(physics_3d[:, :, split_train:split_dev], (2, 0, 1))[..., None].astype(np.float32)
        p_test = np.transpose(physics_3d[:, :, split_dev:], (2, 0, 1))[..., None].astype(np.float32)

    # Do not drop pixels/samples; only sanitize model inputs.
    x_train = np.nan_to_num(x_train, nan=0.0)
    x_val = np.nan_to_num(x_val, nan=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0)
    if p_train is not None:
        p_train = np.nan_to_num(p_train, nan=0.0)
    if p_val is not None:
        p_val = np.nan_to_num(p_val, nan=0.0)
    if p_test is not None:
        p_test = np.nan_to_num(p_test, nan=0.0)

    return x_train, y_train, x_val, y_val, x_test, y_test, p_train, p_val, p_test


def load_from_nc(
    ds: xr.Dataset,
    *,
    feature_keys: list[str],
    target_var: str,
    horizon: int,
    train_frac: float,
    dev_frac: float,
    prediction_mode: str,
    physics_var: str | None,
    rollout_steps: int,
    require_all_features: bool,
    online_physics_coupling: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[str],
]:
    feature_names = resolve_feature_names(ds, feature_keys, require_all=require_all_features)
    if target_var not in ds.data_vars:
        raise KeyError(f"Target variable not found: {target_var}")

    if horizon <= 0:
        raise ValueError("--horizon must be positive")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be in (0, 1)")
    if not (0.0 < dev_frac < 1.0):
        raise ValueError("--dev-frac must be in (0, 1)")

    feat_da = ds[feature_names].to_array("feature").transpose("latitude", "longitude", TIME_DIM, "feature")
    tgt_da = ds[target_var].transpose("latitude", "longitude", TIME_DIM)

    feat = feat_da.values
    tgt = tgt_da.values
    if feat.shape[2] <= horizon:
        raise ValueError(f"Time size {feat.shape[2]} must be larger than horizon {horizon}")

    phys_src = None
    if prediction_mode == "delta":
        raise ValueError("D5 is pure-DL. prediction_mode='delta' is disabled.")

    if prediction_mode == "trend_sum":
        if rollout_steps <= 0:
            raise ValueError("--rollout-steps must be positive.")

        # Multi-task auxiliary supervision channels in this order:
        # [u10_next, v10_next, t2m_next, ceres_next, rh_next]
        u10_idx = find_feature_index(feature_names, "U10", required=True)
        v10_idx = find_feature_index(feature_names, "V10", required=True)
        t2m_idx = find_feature_index(feature_names, "T2M", required=True)
        ceres_idx = find_feature_index(feature_names, "CERES_ALL", required=True)
        rh_idx = find_feature_index(feature_names, "R", required=True)

        t_size = feat.shape[2]
        n_start = t_size - rollout_steps
        if n_start <= 1:
            raise ValueError(
                f"Not enough timesteps ({t_size}) for rollout_steps={rollout_steps}. "
                f"Need at least rollout_steps+2 timesteps."
            )

        # Strict anti-leakage rollout slicing:
        # Anchor at time t, use forcing X_t for all forecast leads t+1..t+K.
        # Dynamic states are iteratively updated in the rollout.
        feat_anchor = feat[:, :, :n_start, :]  # [Lat, Lon, Start, F] = X_t
        x_seq = np.repeat(feat_anchor[:, :, :, None, :], repeats=int(rollout_steps), axis=3)  # [Lat, Lon, Start, K, F]

        # Targets are true AOD states at t+1..t+K.
        y_win = np.lib.stride_tricks.sliding_window_view(tgt, window_shape=rollout_steps, axis=2)
        y_seq = y_win[:, :, 1 : n_start + 1, :][..., None]  # [Lat, Lon, Start, K, 1]

        def _next_seq_from_feature(idx: int) -> np.ndarray:
            src = feat[:, :, :, int(idx)]
            win = np.lib.stride_tricks.sliding_window_view(src, window_shape=rollout_steps, axis=2)
            return win[:, :, 1 : n_start + 1, :][..., None]

        aux_seq = np.concatenate(
            [
                _next_seq_from_feature(int(u10_idx)),
                _next_seq_from_feature(int(v10_idx)),
                _next_seq_from_feature(int(t2m_idx)),
                _next_seq_from_feature(int(ceres_idx)),
                _next_seq_from_feature(int(rh_idx)),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)

        p_seq = None

        split_dev = int(math.floor(n_start * dev_frac))
        if split_dev <= 1 or split_dev >= n_start:
            raise ValueError(
                f"Invalid dev split for trend_sum: split_dev={split_dev}, n_start={n_start}, dev_frac={dev_frac}"
            )
        split_train = int(math.floor(split_dev * train_frac))
        if split_train <= 0 or split_train >= split_dev:
            raise ValueError(
                "Invalid train split in development subset for trend_sum: "
                f"split_train={split_train}, split_dev={split_dev}, train_frac={train_frac}"
            )

        # Keep spatial grid intact for PDE coupling: [Start, K, Lat, Lon, *]
        x_train = np.transpose(x_seq[:, :, :split_train, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        y_train = np.transpose(y_seq[:, :, :split_train, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        aux_train = np.transpose(aux_seq[:, :, :split_train, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        if p_seq is not None:
            p_train = np.transpose(p_seq[:, :, :split_train, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        else:
            p_train = None

        x_val = np.transpose(x_seq[:, :, split_train:split_dev, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        y_val = np.transpose(y_seq[:, :, split_train:split_dev, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        aux_val = np.transpose(aux_seq[:, :, split_train:split_dev, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        if p_seq is not None:
            p_val = np.transpose(p_seq[:, :, split_train:split_dev, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        else:
            p_val = None

        x_test = np.transpose(x_seq[:, :, split_dev:, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        y_test = np.transpose(y_seq[:, :, split_dev:, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        aux_test = np.transpose(aux_seq[:, :, split_dev:, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        if p_seq is not None:
            p_test = np.transpose(p_seq[:, :, split_dev:, :, :], (2, 3, 0, 1, 4)).astype(np.float32)
        else:
            p_test = None

        # Do not drop spatial points: fill NaN only on inputs, keep NaN targets for masked loss.
        x_train = np.nan_to_num(x_train, nan=0.0)
        x_val = np.nan_to_num(x_val, nan=0.0)
        x_test = np.nan_to_num(x_test, nan=0.0)
        if p_train is not None:
            p_train = np.nan_to_num(p_train, nan=0.0)
        if p_val is not None:
            p_val = np.nan_to_num(p_val, nan=0.0)
        if p_test is not None:
            p_test = np.nan_to_num(p_test, nan=0.0)

        return (
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            p_train,
            p_val,
            p_test,
            aux_train,
            aux_val,
            aux_test,
            feature_names,
        )

    feat_h = feat[:, :, :-horizon, :]
    tgt_h = tgt[:, :, horizon:]
    phys_h = None
    x_train, y_train, x_val, y_val, x_test, y_test, p_train, p_val, p_test = _split_keep_grid_time(
        feat_h,
        tgt_h,
        train_frac,
        dev_frac,
        physics_3d=phys_h,
    )
    aux_train = aux_val = aux_test = None
    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        p_train,
        p_val,
        p_test,
        aux_train,
        aux_val,
        aux_test,
        feature_names,
    )
def load_npz(
    npz_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    data = np.load(npz_path)
    required = ("x_train", "y_train", "x_val", "y_val")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing arrays in NPZ: {missing}")
    if "x_test" not in data or "y_test" not in data:
        logging.warning("NPZ missing x_test/y_test; falling back to x_val/y_val as evaluation split.")
        x_test = data["x_val"]
        y_test = data["y_val"]
    else:
        x_test = data["x_test"]
        y_test = data["y_test"]

    p_train = data["p_train"] if "p_train" in data else None
    p_val = data["p_val"] if "p_val" in data else None
    if "p_test" in data:
        p_test = data["p_test"]
    elif "p_val" in data:
        p_test = data["p_val"]
    else:
        p_test = None

    aux_train = data["aux_train"] if "aux_train" in data else None
    aux_val = data["aux_val"] if "aux_val" in data else None
    if "aux_test" in data:
        aux_test = data["aux_test"]
    elif "aux_val" in data:
        aux_test = data["aux_val"]
    else:
        aux_test = None

    return (
        np.asarray(data["x_train"], dtype=np.float32),
        np.asarray(data["y_train"], dtype=np.float32),
        np.asarray(data["x_val"], dtype=np.float32),
        np.asarray(data["y_val"], dtype=np.float32),
        np.asarray(x_test, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        (np.asarray(p_train, dtype=np.float32) if p_train is not None else None),
        (np.asarray(p_val, dtype=np.float32) if p_val is not None else None),
        (np.asarray(p_test, dtype=np.float32) if p_test is not None else None),
        (np.asarray(aux_train, dtype=np.float32) if aux_train is not None else None),
        (np.asarray(aux_val, dtype=np.float32) if aux_val is not None else None),
        (np.asarray(aux_test, dtype=np.float32) if aux_test is not None else None),
    )


def _unpack_batch(
    batch: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 2:
        features, target = batch
        return features, target, None, None
    if len(batch) == 3:
        features, target, third = batch
        # trend_sum physics shape last channel == 1; auxiliary tasks use 5 channels.
        if third.shape[-1] == 1:
            return features, target, third, None
        return features, target, None, third
    if len(batch) == 4:
        features, target, aod_phys, aux_state = batch
        return features, target, aod_phys, aux_state
    raise ValueError(f"Unexpected batch size {len(batch)} from dataset.")
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_cfg: LossConfig,
    prediction_mode: str,
    aod_feature_index: int | None,
    dt_days: float,
    rollout_decay: float,
    max_points_per_forward: int,
    online_physics_coupling: bool,
    iterative_aux_forcing: bool,
    learn_diffusion_coeff: bool,
    u10_feature_index: int | None,
    v10_feature_index: int | None,
    wind_speed_feature_index: int | None,
    t2m_feature_index: int | None,
    ceres_feature_index: int | None,
    rh_feature_index: int | None,
    physics_grid: object | None,
    physics_cfg: object | None,
    diffusion_task_model: nn.Module | None,
    diffusion_k_min: float,
    diffusion_k_base: float,
    diffusion_k_slope: float,
    diffusion_k_cap: float,
) -> tuple[float, float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    if diffusion_task_model is not None:
        diffusion_task_model.train(train_mode)

    total_loss = 0.0
    total_mse = 0.0
    total_n = 0
    total_phys_trend_mean = 0.0
    total_dl_trend_mean = 0.0
    total_total_trend_mean = 0.0
    total_rad_trend_mean = 0.0
    total_dry_trend_mean = 0.0
    total_wet_trend_mean = 0.0
    total_k_dry_mean = 0.0
    total_k_wet_mean = 0.0
    total_kappa_diff_mean = 0.0
    total_sparsity_penalty = 0.0
    step_phys_trend_sum: list[float] | None = None
    step_dl_trend_sum: list[float] | None = None
    step_total_trend_sum: list[float] | None = None
    step_rad_trend_sum: list[float] | None = None
    step_dry_trend_sum: list[float] | None = None
    step_wet_trend_sum: list[float] | None = None
    step_k_dry_sum: list[float] | None = None
    step_k_wet_sum: list[float] | None = None
    step_kappa_diff_sum: list[float] | None = None
    step_sparsity_sum: list[float] | None = None
    step_pred_state_sum: list[float] | None = None
    step_target_state_sum: list[float] | None = None
    step_mse_sum: list[float] | None = None
    has_trend_stats = False
    warned_no_grad = False

    for batch in loader:
        features, target, aod_phys, aux_state = _unpack_batch(batch)

        features = features.to(device)
        target = target.to(device)
        if target.ndim == 1:
            target = target.unsqueeze(1)
        if aod_phys is not None:
            aod_phys = aod_phys.to(device)
            if aod_phys.ndim == 1:
                aod_phys = aod_phys.unsqueeze(1)
        if aux_state is not None:
            aux_state = aux_state.to(device)

        if prediction_mode == "trend_sum":
            if aod_feature_index is None:
                raise ValueError("trend_sum mode requires valid aod_feature_index.")
            loss, _, parts = compute_autoregressive_trend_loss(
                model=model,
                feature_seq=features,
                target_state_seq=target,
                physics_trend_seq=aod_phys,
                auxiliary_state_seq=aux_state,
                aod_feature_index=aod_feature_index,
                u10_feature_index=u10_feature_index,
                v10_feature_index=v10_feature_index,
                ws_feature_index=wind_speed_feature_index,
                t2m_feature_index=t2m_feature_index,
                ceres_feature_index=ceres_feature_index,
                rh_feature_index=rh_feature_index,
                dt_days=dt_days,
                time_decay=rollout_decay,
                max_points_per_forward=max_points_per_forward,
                cfg=loss_cfg,
                online_physics_coupling=bool(online_physics_coupling),
                iterative_aux_forcing=bool(iterative_aux_forcing),
                learn_diffusion_coeff=bool(learn_diffusion_coeff),
                physics_grid=physics_grid,
                physics_cfg=physics_cfg,
                diffusion_task_model=diffusion_task_model,
                diffusion_k_min=float(diffusion_k_min),
                diffusion_k_base=float(diffusion_k_base),
                diffusion_k_slope=float(diffusion_k_slope),
                diffusion_k_cap=float(diffusion_k_cap),
            )
        else:
            dl_out = model(features)
            loss, _, parts = compute_coupled_loss(
                dl_output=dl_out,
                target=target,
                aod_phys=aod_phys,
                cfg=loss_cfg,
                prediction_mode=prediction_mode,
            )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            elif not warned_no_grad:
                logging.warning(
                    "Loss has no autograd graph (physics-core-only mode). "
                    "Skipping backward/optimizer step."
                )
                warned_no_grad = True

        n = features.shape[0]
        total_n += n
        total_loss += float(loss.detach().item()) * n
        total_mse += float(parts["mse"]) * n
        if "phys_trend_mean" in parts:
            has_trend_stats = True
            total_phys_trend_mean += float(parts["phys_trend_mean"]) * n
            total_dl_trend_mean += float(parts["dl_trend_mean"]) * n
            total_total_trend_mean += float(parts["total_trend_mean"]) * n
            total_rad_trend_mean += float(parts.get("rad_trend_mean", 0.0)) * n
            total_dry_trend_mean += float(parts.get("dry_trend_mean", 0.0)) * n
            total_wet_trend_mean += float(parts.get("wet_trend_mean", 0.0)) * n
            total_k_dry_mean += float(parts.get("k_dry_mean", 0.0)) * n
            total_k_wet_mean += float(parts.get("k_wet_mean", 0.0)) * n
            total_kappa_diff_mean += float(parts.get("kappa_diff_mean", 0.0)) * n
            total_sparsity_penalty += float(parts.get("sparsity_penalty", 0.0)) * n
            pred_steps = parts.get("pred_state_step_means")
            target_steps = parts.get("target_state_step_means")
            phys_steps = parts.get("phys_trend_step_means")
            dl_steps = parts.get("dl_trend_step_means")
            total_steps = parts.get("total_trend_step_means")
            rad_steps = parts.get("rad_trend_step_means")
            dry_steps = parts.get("dry_trend_step_means")
            wet_steps = parts.get("wet_trend_step_means")
            k_dry_steps = parts.get("k_dry_step_means")
            k_wet_steps = parts.get("k_wet_step_means")
            kappa_steps = parts.get("kappa_diff_step_means")
            sparsity_steps = parts.get("sparsity_step_means")
            mse_steps = parts.get("mse_step_means")
            if isinstance(pred_steps, list) and isinstance(target_steps, list):
                if step_pred_state_sum is None:
                    step_pred_state_sum = [0.0] * len(pred_steps)
                    step_target_state_sum = [0.0] * len(target_steps)
                for i, val in enumerate(pred_steps):
                    step_pred_state_sum[i] += float(val) * n
                for i, val in enumerate(target_steps):
                    step_target_state_sum[i] += float(val) * n
            if isinstance(phys_steps, list) and isinstance(dl_steps, list) and isinstance(total_steps, list):
                if step_phys_trend_sum is None:
                    step_phys_trend_sum = [0.0] * len(phys_steps)
                    step_dl_trend_sum = [0.0] * len(dl_steps)
                    step_total_trend_sum = [0.0] * len(total_steps)
                for i, val in enumerate(phys_steps):
                    step_phys_trend_sum[i] += float(val) * n
                for i, val in enumerate(dl_steps):
                    step_dl_trend_sum[i] += float(val) * n
                for i, val in enumerate(total_steps):
                    step_total_trend_sum[i] += float(val) * n
            if (
                isinstance(rad_steps, list)
                and isinstance(dry_steps, list)
                and isinstance(wet_steps, list)
                and isinstance(k_dry_steps, list)
                and isinstance(k_wet_steps, list)
            ):
                if step_rad_trend_sum is None:
                    step_rad_trend_sum = [0.0] * len(rad_steps)
                    step_dry_trend_sum = [0.0] * len(dry_steps)
                    step_wet_trend_sum = [0.0] * len(wet_steps)
                    step_k_dry_sum = [0.0] * len(k_dry_steps)
                    step_k_wet_sum = [0.0] * len(k_wet_steps)
                for i, val in enumerate(rad_steps):
                    step_rad_trend_sum[i] += float(val) * n
                for i, val in enumerate(dry_steps):
                    step_dry_trend_sum[i] += float(val) * n
                for i, val in enumerate(wet_steps):
                    step_wet_trend_sum[i] += float(val) * n
                for i, val in enumerate(k_dry_steps):
                    step_k_dry_sum[i] += float(val) * n
                for i, val in enumerate(k_wet_steps):
                    step_k_wet_sum[i] += float(val) * n
            if isinstance(kappa_steps, list):
                if step_kappa_diff_sum is None:
                    step_kappa_diff_sum = [0.0] * len(kappa_steps)
                for i, val in enumerate(kappa_steps):
                    if math.isfinite(float(val)):
                        step_kappa_diff_sum[i] += float(val) * n
            if isinstance(sparsity_steps, list):
                if step_sparsity_sum is None:
                    step_sparsity_sum = [0.0] * len(sparsity_steps)
                for i, val in enumerate(sparsity_steps):
                    step_sparsity_sum[i] += float(val) * n
            if isinstance(mse_steps, list):
                if step_mse_sum is None:
                    step_mse_sum = [0.0] * len(mse_steps)
                for i, val in enumerate(mse_steps):
                    step_mse_sum[i] += float(val) * n

    denom = max(total_n, 1)
    extra = {}
    if has_trend_stats:
        extra = {
            "phys_trend_mean": total_phys_trend_mean / denom,
            "dl_trend_mean": total_dl_trend_mean / denom,
            "total_trend_mean": total_total_trend_mean / denom,
            "rad_trend_mean": total_rad_trend_mean / denom,
            "dry_trend_mean": total_dry_trend_mean / denom,
            "wet_trend_mean": total_wet_trend_mean / denom,
            "k_dry_mean": total_k_dry_mean / denom,
            "k_wet_mean": total_k_wet_mean / denom,
            "kappa_diff_mean": total_kappa_diff_mean / denom,
            "sparsity_penalty": total_sparsity_penalty / denom,
        }
        if step_pred_state_sum is not None and step_target_state_sum is not None:
            extra["pred_state_step_means"] = [v / denom for v in step_pred_state_sum]
            extra["target_state_step_means"] = [v / denom for v in step_target_state_sum]
        if step_phys_trend_sum is not None and step_dl_trend_sum is not None and step_total_trend_sum is not None:
            extra["phys_trend_step_means"] = [v / denom for v in step_phys_trend_sum]
            extra["dl_trend_step_means"] = [v / denom for v in step_dl_trend_sum]
            extra["total_trend_step_means"] = [v / denom for v in step_total_trend_sum]
        if (
            step_rad_trend_sum is not None
            and step_dry_trend_sum is not None
            and step_wet_trend_sum is not None
            and step_k_dry_sum is not None
            and step_k_wet_sum is not None
        ):
            extra["rad_trend_step_means"] = [v / denom for v in step_rad_trend_sum]
            extra["dry_trend_step_means"] = [v / denom for v in step_dry_trend_sum]
            extra["wet_trend_step_means"] = [v / denom for v in step_wet_trend_sum]
            extra["k_dry_step_means"] = [v / denom for v in step_k_dry_sum]
            extra["k_wet_step_means"] = [v / denom for v in step_k_wet_sum]
        if step_kappa_diff_sum is not None:
            extra["kappa_diff_step_means"] = [v / denom for v in step_kappa_diff_sum]
        if step_sparsity_sum is not None:
            extra["sparsity_step_means"] = [v / denom for v in step_sparsity_sum]
        if step_mse_sum is not None:
            extra["mse_step_means"] = [v / denom for v in step_mse_sum]
    return total_loss / denom, total_mse / denom, extra

def _accumulate_masked_regression_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    sums: dict[str, float],
) -> None:
    valid_mask = torch.isfinite(target)
    if not valid_mask.any():
        return
    y_pred = pred[valid_mask]
    y_true = target[valid_mask]
    diff = y_pred - y_true

    sums["count"] += float(y_true.numel())
    sums["sse"] += float(torch.sum(diff * diff).detach().item())
    sums["sae"] += float(torch.sum(torch.abs(diff)).detach().item())
    sums["sum_y"] += float(torch.sum(y_true).detach().item())
    sums["sum_y2"] += float(torch.sum(y_true * y_true).detach().item())


def evaluate_regression_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    prediction_mode: str,
    aod_feature_index: int | None,
    dt_days: float,
    rollout_decay: float,
    max_points_per_forward: int,
    loss_cfg: LossConfig,
    online_physics_coupling: bool,
    iterative_aux_forcing: bool,
    learn_diffusion_coeff: bool,
    u10_feature_index: int | None,
    v10_feature_index: int | None,
    wind_speed_feature_index: int | None,
    t2m_feature_index: int | None,
    ceres_feature_index: int | None,
    rh_feature_index: int | None,
    physics_grid: object | None,
    physics_cfg: object | None,
    diffusion_task_model: nn.Module | None,
    diffusion_k_min: float,
    diffusion_k_base: float,
    diffusion_k_slope: float,
    diffusion_k_cap: float,
    aod_target_min: float | None = None,
    aod_target_range: float | None = None,
    aux_denorm_params: dict[str, tuple[float | None, float | None]] | None = None,
    aux_denorm_sources: dict[str, str] | None = None,
) -> dict[str, float | list[float] | str]:
    def _new_sums() -> dict[str, float]:
        return {"count": 0.0, "sse": 0.0, "sae": 0.0, "sum_y": 0.0, "sum_y2": 0.0}

    def _finalize(sums: dict[str, float]) -> dict[str, float]:
        count = max(sums["count"], 1.0)
        rmse = math.sqrt(sums["sse"] / count)
        mae = sums["sae"] / count
        y_mean = sums["sum_y"] / count
        sst = sums["sum_y2"] - 2.0 * y_mean * sums["sum_y"] + count * y_mean * y_mean
        if sst <= 1e-12:
            r2 = float("nan")
        else:
            r2 = 1.0 - (sums["sse"] / sst)
        return {
            "r2": float(r2),
            "rmse": float(rmse),
            "mae": float(mae),
            "n_valid": float(sums["count"]),
        }

    model.eval()
    if diffusion_task_model is not None:
        diffusion_task_model.eval()

    sums = _new_sums()
    step_sums: list[dict[str, float]] | None = None

    aux_names = ("t2m", "ceres", "rh")
    aux_denorm_params = aux_denorm_params or {}
    aux_denorm_sources = aux_denorm_sources or {}

    def _aux_denorm_pair(name: str) -> tuple[float, float] | None:
        pair = aux_denorm_params.get(name)
        if pair is None:
            return None
        mn, rg = pair
        if mn is None or rg is None:
            return None
        mn_f = float(mn)
        rg_f = float(rg)
        if not (np.isfinite(mn_f) and np.isfinite(rg_f)):
            return None
        if rg_f <= 1e-12:
            return None
        return mn_f, rg_f
    aux_sums: dict[str, dict[str, float]] = {name: _new_sums() for name in aux_names}
    aux_step_sums: dict[str, list[dict[str, float]] | None] = {name: None for name in aux_names}
    warned_aux_eval_fallback = False

    with torch.no_grad():
        for batch in loader:
            features, target, aod_phys, aux_state = _unpack_batch(batch)

            features = features.to(device)
            target = target.to(device)
            if aod_phys is not None:
                aod_phys = aod_phys.to(device)
            if aux_state is not None:
                aux_state = aux_state.to(device)

            pred_parts: dict[str, object] | None = None
            if prediction_mode == "trend_sum":
                if aod_feature_index is None:
                    raise ValueError("trend_sum metrics require aod_feature_index.")
                trend_kwargs = dict(
                    model=model,
                    feature_seq=features,
                    target_state_seq=target,
                    physics_trend_seq=aod_phys,
                    auxiliary_state_seq=aux_state,
                    aod_feature_index=aod_feature_index,
                    u10_feature_index=u10_feature_index,
                    v10_feature_index=v10_feature_index,
                    ws_feature_index=wind_speed_feature_index,
                    t2m_feature_index=t2m_feature_index,
                    ceres_feature_index=ceres_feature_index,
                    rh_feature_index=rh_feature_index,
                    dt_days=dt_days,
                    time_decay=rollout_decay,
                    max_points_per_forward=max_points_per_forward,
                    cfg=loss_cfg,
                    online_physics_coupling=bool(online_physics_coupling),
                    iterative_aux_forcing=bool(iterative_aux_forcing),
                    learn_diffusion_coeff=bool(learn_diffusion_coeff),
                    physics_grid=physics_grid,
                    physics_cfg=physics_cfg,
                    diffusion_task_model=diffusion_task_model,
                    diffusion_k_min=float(diffusion_k_min),
                    diffusion_k_base=float(diffusion_k_base),
                    diffusion_k_slope=float(diffusion_k_slope),
                    diffusion_k_cap=float(diffusion_k_cap),
                )
                trend_kwargs["return_aux_predictions"] = True
                try:
                    _, pred, pred_parts = compute_autoregressive_trend_loss(**trend_kwargs)
                except TypeError as exc:
                    if "return_aux_predictions" not in str(exc):
                        raise
                    trend_kwargs.pop("return_aux_predictions", None)
                    _, pred, pred_parts = compute_autoregressive_trend_loss(**trend_kwargs)
                    if not warned_aux_eval_fallback:
                        logging.warning(
                            "compute_autoregressive_trend_loss() in current losses.py does not support return_aux_predictions; "
                            "AUX eval metrics are skipped unless losses.py is updated."
                        )
                        warned_aux_eval_fallback = True
            else:
                dl_out = model(features)
                _, pred, _ = compute_coupled_loss(
                    dl_output=dl_out,
                    target=target,
                    aod_phys=aod_phys,
                    cfg=loss_cfg,
                    prediction_mode=prediction_mode,
                )

            if aod_target_min is not None and aod_target_range is not None:
                pred = pred * aod_target_range + aod_target_min
                target = target * aod_target_range + aod_target_min

            _accumulate_masked_regression_sums(pred, target, sums)
            if prediction_mode == "trend_sum" and pred.ndim >= 2 and target.ndim >= 2:
                step_n = min(int(pred.shape[1]), int(target.shape[1]))
                if step_sums is None:
                    step_sums = [_new_sums() for _ in range(step_n)]
                for step_idx in range(step_n):
                    _accumulate_masked_regression_sums(
                        pred[:, step_idx, ...],
                        target[:, step_idx, ...],
                        step_sums[step_idx],
                    )

            if prediction_mode == "trend_sum" and aux_state is not None and pred_parts is not None:
                aux_pred_map: dict[str, torch.Tensor | None] = {
                    "t2m": pred_parts.get("pred_t2m_state_seq"),
                    "ceres": pred_parts.get("pred_ceres_state_seq"),
                    "rh": pred_parts.get("pred_rh_state_seq"),
                }
                aux_true_map: dict[str, torch.Tensor] = {
                    "t2m": aux_state[..., 2:3],
                    "ceres": aux_state[..., 3:4],
                    "rh": aux_state[..., 4:5],
                }
                for name in aux_names:
                    pred_aux = aux_pred_map[name]
                    if not isinstance(pred_aux, torch.Tensor):
                        continue
                    true_aux = aux_true_map[name]
                    denorm_pair = _aux_denorm_pair(name)
                    if denorm_pair is not None:
                        mn_f, rg_f = denorm_pair
                        pred_aux_eval = pred_aux * rg_f + mn_f
                        true_aux_eval = true_aux * rg_f + mn_f
                    else:
                        pred_aux_eval = pred_aux
                        true_aux_eval = true_aux
                    _accumulate_masked_regression_sums(pred_aux_eval, true_aux_eval, aux_sums[name])
                    if pred_aux_eval.ndim >= 2 and true_aux_eval.ndim >= 2:
                        step_n_aux = min(int(pred_aux_eval.shape[1]), int(true_aux_eval.shape[1]))
                        if aux_step_sums[name] is None:
                            aux_step_sums[name] = [_new_sums() for _ in range(step_n_aux)]
                        assert aux_step_sums[name] is not None
                        for step_idx in range(step_n_aux):
                            _accumulate_masked_regression_sums(
                                pred_aux_eval[:, step_idx, ...],
                                true_aux_eval[:, step_idx, ...],
                                aux_step_sums[name][step_idx],
                            )

    aod_metrics = _finalize(sums)
    metrics: dict[str, float | list[float]] = dict(aod_metrics)

    if step_sums is not None:
        r2_step: list[float] = []
        rmse_step: list[float] = []
        mae_step: list[float] = []
        n_valid_step: list[float] = []
        for step_sum in step_sums:
            step_metric = _finalize(step_sum)
            r2_step.append(float(step_metric["r2"]))
            rmse_step.append(float(step_metric["rmse"]))
            mae_step.append(float(step_metric["mae"]))
            n_valid_step.append(float(step_metric["n_valid"]))
        metrics["r2_step"] = r2_step
        metrics["rmse_step"] = rmse_step
        metrics["mae_step"] = mae_step
        metrics["n_valid_step"] = n_valid_step

    for name in aux_names:
        count_val = float(aux_sums[name]["count"])
        metrics[f"{name}_metric_space"] = "physical_denormalized" if _aux_denorm_pair(name) is not None else "normalized"
        metrics[f"{name}_denorm_source"] = str(aux_denorm_sources.get(name, "not_found"))

        if count_val <= 0.0:
            metrics[f"{name}_r2"] = float("nan")
            metrics[f"{name}_rmse"] = float("nan")
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_n_valid"] = 0.0
            metrics[f"{name}_eval_note"] = "no_valid_samples_or_missing_predictions"
            continue

        m = _finalize(aux_sums[name])
        metrics[f"{name}_r2"] = float(m["r2"])
        metrics[f"{name}_rmse"] = float(m["rmse"])
        metrics[f"{name}_mae"] = float(m["mae"])
        metrics[f"{name}_n_valid"] = float(m["n_valid"])
        metrics[f"{name}_eval_note"] = "ok"

        step_bucket = aux_step_sums[name]
        if step_bucket is not None:
            r2_step: list[float] = []
            rmse_step: list[float] = []
            mae_step: list[float] = []
            n_valid_step: list[float] = []
            for step_sum in step_bucket:
                step_metric = _finalize(step_sum)
                r2_step.append(float(step_metric["r2"]))
                rmse_step.append(float(step_metric["rmse"]))
                mae_step.append(float(step_metric["mae"]))
                n_valid_step.append(float(step_metric["n_valid"]))
            metrics[f"{name}_r2_step"] = r2_step
            metrics[f"{name}_rmse_step"] = rmse_step
            metrics[f"{name}_mae_step"] = mae_step
            metrics[f"{name}_n_valid_step"] = n_valid_step

    return metrics


def collect_regression_outputs(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    prediction_mode: str,
    aod_feature_index: int | None,
    dt_days: float,
    rollout_decay: float,
    max_points_per_forward: int,
    loss_cfg: LossConfig,
    online_physics_coupling: bool,
    iterative_aux_forcing: bool,
    learn_diffusion_coeff: bool,
    u10_feature_index: int | None,
    v10_feature_index: int | None,
    wind_speed_feature_index: int | None,
    t2m_feature_index: int | None,
    ceres_feature_index: int | None,
    rh_feature_index: int | None,
    physics_grid: object | None,
    physics_cfg: object | None,
    diffusion_task_model: nn.Module | None,
    diffusion_k_min: float,
    diffusion_k_base: float,
    diffusion_k_slope: float,
    diffusion_k_cap: float,
    aod_target_min: float | None = None,
    aod_target_range: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    if diffusion_task_model is not None:
        diffusion_task_model.eval()
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features, target, aod_phys, aux_state = _unpack_batch(batch)

            features = features.to(device)
            target = target.to(device)
            if aod_phys is not None:
                aod_phys = aod_phys.to(device)
            if aux_state is not None:
                aux_state = aux_state.to(device)

            if prediction_mode == "trend_sum":
                if aod_feature_index is None:
                    raise ValueError("trend_sum output collection requires aod_feature_index.")
                _, pred, _ = compute_autoregressive_trend_loss(
                    model=model,
                    feature_seq=features,
                    target_state_seq=target,
                    physics_trend_seq=aod_phys,
                    auxiliary_state_seq=aux_state,
                    aod_feature_index=aod_feature_index,
                    u10_feature_index=u10_feature_index,
                    v10_feature_index=v10_feature_index,
                    ws_feature_index=wind_speed_feature_index,
                    t2m_feature_index=t2m_feature_index,
                    ceres_feature_index=ceres_feature_index,
                    rh_feature_index=rh_feature_index,
                    dt_days=dt_days,
                    time_decay=rollout_decay,
                    max_points_per_forward=max_points_per_forward,
                    cfg=loss_cfg,
                    online_physics_coupling=bool(online_physics_coupling),
                    iterative_aux_forcing=bool(iterative_aux_forcing),
                    learn_diffusion_coeff=bool(learn_diffusion_coeff),
                    physics_grid=physics_grid,
                    physics_cfg=physics_cfg,
                    diffusion_task_model=diffusion_task_model,
                    diffusion_k_min=float(diffusion_k_min),
                    diffusion_k_base=float(diffusion_k_base),
                    diffusion_k_slope=float(diffusion_k_slope),
                    diffusion_k_cap=float(diffusion_k_cap),
                )
            else:
                dl_out = model(features)
                _, pred, _ = compute_coupled_loss(
                    dl_output=dl_out,
                    target=target,
                    aod_phys=aod_phys,
                    cfg=loss_cfg,
                    prediction_mode=prediction_mode,
                )

            if aod_target_min is not None and aod_target_range is not None:
                pred = pred * aod_target_range + aod_target_min
                target = target * aod_target_range + aod_target_min

            pred_batches.append(pred.detach().cpu().numpy().astype(np.float32))
            target_batches.append(target.detach().cpu().numpy().astype(np.float32))

    if not pred_batches:
        empty = np.zeros((0, 1), dtype=np.float32)
        return empty, empty
    return np.concatenate(pred_batches, axis=0), np.concatenate(target_batches, axis=0)


def save_regression_outputs_to_nc(
    *,
    out_path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    prediction_mode: str,
    metric_space: str,
    online_physics_coupling: bool,
) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

    if pred.ndim == 5:
        dims = ("sample", "step", "latitude", "longitude", "channel")
    elif pred.ndim == 4:
        dims = ("sample", "latitude", "longitude", "channel")
    elif pred.ndim == 3:
        dims = ("sample", "point", "channel")
    elif pred.ndim == 2:
        dims = ("sample", "channel")
    else:
        dims = tuple(f"dim_{i}" for i in range(pred.ndim))

    valid_mask = np.isfinite(target)
    bias = (pred - target).astype(np.float32, copy=False)
    abs_error = np.abs(bias).astype(np.float32, copy=False)
    sq_error = (bias * bias).astype(np.float32, copy=False)
    bias_masked = np.where(valid_mask, bias, np.nan).astype(np.float32, copy=False)
    abs_error_masked = np.where(valid_mask, abs_error, np.nan).astype(np.float32, copy=False)
    sq_error_masked = np.where(valid_mask, sq_error, np.nan).astype(np.float32, copy=False)

    ds_out = xr.Dataset(
        data_vars={
            "A_hat": (dims, pred),
            "target": (dims, target),
            "valid_mask": (dims, valid_mask),
            "bias": (dims, bias),
            "bias_masked": (dims, bias_masked),
            "abs_error": (dims, abs_error),
            "abs_error_masked": (dims, abs_error_masked),
            "sq_error": (dims, sq_error),
            "sq_error_masked": (dims, sq_error_masked),
        },
        attrs={
            "prediction_mode": str(prediction_mode),
            "metric_space": str(metric_space),
            "online_physics_coupling": int(bool(online_physics_coupling)),
            "bias_definition": "bias = A_hat - target",
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)

def _format_state_compare(extra: dict[str, float]) -> str:
    pred_steps = extra.get("pred_state_step_means", [])
    target_steps = extra.get("target_state_step_means", [])
    pred_line = " | ".join(f"t+{i+1}:{v:.6e}" for i, v in enumerate(pred_steps))
    target_line = " | ".join(f"t+{i+1}:{v:.6e}" for i, v in enumerate(target_steps))
    return f"state_pred   : {pred_line}\nstate_target : {target_line}"


def _format_trend_compare(extra: dict[str, float]) -> str:
    phys_steps = extra.get("phys_trend_step_means", [])
    dl_steps = extra.get("dl_trend_step_means", [])
    total_steps = extra.get("total_trend_step_means", [])
    lines = []
    for i in range(min(len(phys_steps), len(dl_steps), len(total_steps))):
        lines.append(
            f"trend_step_{i+1}: phys={phys_steps[i]:.6e} + dl={dl_steps[i]:.6e} = total={total_steps[i]:.6e}"
        )
    return "\n".join(lines)


def _format_process_compare(extra: dict[str, float]) -> str:
    rad_steps = extra.get("rad_trend_step_means", [])
    dep_steps = extra.get("dep_trend_step_means", extra.get("dry_trend_step_means", []))
    k_dep_steps = extra.get("k_dep_step_means", extra.get("k_dry_step_means", []))
    lines = []
    for i in range(min(len(rad_steps), len(dep_steps), len(k_dep_steps))):
        lines.append(
            "dl_step_%d: rad=%.6e dep=%.6e | k_dep=%.6e"
            % (
                i + 1,
                float(rad_steps[i]),
                float(dep_steps[i]),
                float(k_dep_steps[i]),
            )
        )
    return "\n".join(lines)


def _format_step_mse(extra: dict[str, float]) -> str:
    mse_steps = extra.get("mse_step_means", [])
    mse_line = " | ".join(f"t+{i+1}:{v:.6e}" for i, v in enumerate(mse_steps))
    return f"rollout_mse  : {mse_line}"


def _format_step_eval_metrics(metrics: dict[str, float | list[float]]) -> str:
    r2_steps = metrics.get("r2_step", [])
    rmse_steps = metrics.get("rmse_step", [])
    mae_steps = metrics.get("mae_step", [])
    n_steps = metrics.get("n_valid_step", [])
    if not (
        isinstance(r2_steps, list)
        and isinstance(rmse_steps, list)
        and isinstance(mae_steps, list)
        and isinstance(n_steps, list)
    ):
        return ""
    lines = []
    for i in range(min(len(r2_steps), len(rmse_steps), len(mae_steps), len(n_steps))):
        lines.append(
            "eval_step_%d | R2=%.6f RMSE=%.6f MAE=%.6f N_valid=%.0f"
            % (i + 1, float(r2_steps[i]), float(rmse_steps[i]), float(mae_steps[i]), float(n_steps[i]))
        )
    return "\n".join(lines)


def _run_single(args: argparse.Namespace) -> int:
    _set_global_seed(int(args.seed))
    logging.info("Using seed=%d", int(args.seed))
    if FORCE_ZERO_DL_OUTPUT:
        logging.warning("FORCE_ZERO_DL_OUTPUT=True. DL outputs are disabled.")

    if args.prediction_mode == "trend_sum":
        if bool(args.online_physics_coupling):
            logging.warning("D5 is pure-DL. Overriding --online-physics-coupling to false.")
        args.online_physics_coupling = False
        if bool(args.predict_uv10):
            logging.warning("Ignoring --predict-uv10: UV next-state prediction task has been removed.")
            args.predict_uv10 = False
        if args.physics_var is not None:
            logging.warning("D5 is pure-DL. Ignoring --physics-var=%s.", str(args.physics_var))
        args.physics_var = None
        if bool(args.learn_diffusion_coeff):
            logging.warning("D5 is pure-DL. Overriding --learn-diffusion-coeff to false.")
        args.learn_diffusion_coeff = False

    used_features: list[str] = []
    source_paths: list[str] = []
    stats_paths: list[Path] = []
    lat_values: np.ndarray | None = None
    lon_values: np.ndarray | None = None
    aux_train = aux_val = aux_test = None

    if args.data_npz is not None:
        x_train, y_train, x_val, y_val, x_test, y_test, p_train, p_val, p_test, aux_train, aux_val, aux_test = load_npz(args.data_npz)
        source_paths = [str(args.data_npz)]
        stats_paths = []
    else:
        if args.data_nc is not None:
            if not args.data_nc.exists():
                raise FileNotFoundError(f"NetCDF file not found: {args.data_nc}")
            ds = load_combined_normalized_dataset([args.data_nc])
            source_paths = [str(args.data_nc)]
            stats_paths = _resolve_stats_paths_from_sources(source_paths)
        elif args.data_root is not None:
            filename = TRAIN_SETTINGS["input_filename"]
            norm_paths = resolve_data_paths(args.data_root, args.years, filename)
            ds = load_combined_normalized_dataset(norm_paths)
            source_paths = [str(p) for p in norm_paths]
            stats_paths = _resolve_stats_paths_from_sources(source_paths)
        else:
            raise ValueError("No valid data source found. Set data path in settings.py or provide CLI path.")
        try:
            if "latitude" in ds.coords:
                lat_values = np.asarray(ds["latitude"].values, dtype=np.float32).copy()
            if "longitude" in ds.coords:
                lon_values = np.asarray(ds["longitude"].values, dtype=np.float32).copy()
            x_train, y_train, x_val, y_val, x_test, y_test, p_train, p_val, p_test, aux_train, aux_val, aux_test, used_features = load_from_nc(
                ds,
                feature_keys=list(args.feature_keys),
                target_var=args.target_var,
                horizon=args.horizon,
                train_frac=args.train_frac,
                dev_frac=args.dev_frac,
                prediction_mode=args.prediction_mode,
                physics_var=args.physics_var,
                rollout_steps=args.rollout_steps,
                require_all_features=args.require_all_features,
                online_physics_coupling=bool(args.online_physics_coupling),
            )
        finally:
            ds.close()
    # D5 hard policy: ignore any physics arrays from source data.
    p_train = p_val = p_test = None
    if args.prediction_mode == "trend_sum":
        # D5 design: no auxiliary prediction tasks.
        aux_train = aux_val = aux_test = None

    logging.info("Data source: %s", ", ".join(source_paths))
    logging.info(
        "Split policy | dev_frac=%.2f (front for train+val), train_frac_in_dev=%.2f, test_frac=%.2f (tail)",
        args.dev_frac,
        args.train_frac,
        1.0 - args.dev_frac,
    )
    logging.info(
        "Shapes | x_train=%s y_train=%s x_val=%s y_val=%s x_test=%s y_test=%s",
        tuple(x_train.shape),
        tuple(y_train.shape),
        tuple(x_val.shape),
        tuple(y_val.shape),
        tuple(x_test.shape),
        tuple(y_test.shape),
    )
    if used_features:
        logging.info("Feature keys: %s", ", ".join(args.feature_keys))
        logging.info("Resolved feature names: %s", ", ".join(used_features))
    aod_feature_index = None
    u10_feature_index: int | None = None
    v10_feature_index: int | None = None
    wind_speed_feature_index: int | None = None
    t2m_feature_index: int | None = None
    ceres_feature_index: int | None = None
    rh_feature_index: int | None = None
    physics_grid_t: object | None = None
    physics_cfg_t: object | None = None
    diffusion_task_model: nn.Module | None = None
    if used_features:
        aod_feature_index = find_feature_index(used_features, "AOD", required=False)
    if args.prediction_mode == "trend_sum" and aod_feature_index is None:
        raise ValueError("trend_sum requires 'AOD' feature in inputs.")
    if args.prediction_mode == "trend_sum":
        wind_speed_feature_index = find_feature_index(used_features, "WS", required=False)
        u10_feature_index = find_feature_index(used_features, "U10", required=False)
        v10_feature_index = find_feature_index(used_features, "V10", required=False)
        t2m_feature_index = find_feature_index(used_features, "T2M", required=False)
        ceres_feature_index = find_feature_index(used_features, "CERES_ALL", required=False)
        rh_feature_index = find_feature_index(used_features, "R", required=False)
        logging.info(
            "Autoregressive rollout enabled: rollout_steps=%d, dt_days=%.3f, rollout_decay=%.3f",
            args.rollout_steps,
            args.dt_days,
            args.rollout_decay,
        )
        logging.info("D5 mode: single MLP trend rollout (no physics coupling, no auxiliary-task supervision).")

    if args.prediction_mode == "trend_sum" and args.batch_size > int(TRAIN_SETTINGS.get("batch_size_trend_sum", 1)):
        logging.warning(
            "trend_sum mode is memory heavy. Overriding batch_size %d -> %d. "
            "Change TRAIN_SETTINGS['batch_size_trend_sum'] if needed.",
            args.batch_size,
            int(TRAIN_SETTINGS.get("batch_size_trend_sum", 1)),
        )
        args.batch_size = int(TRAIN_SETTINGS.get("batch_size_trend_sum", 1))

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        torch.cuda.set_device(args.cuda_device_id)
        device = torch.device(f"cuda:{args.cuda_device_id}")
    else:
        device = torch.device("cpu")

    input_dim = int(x_train.shape[-1])
    model = GridPointMLP(
        MLPConfig(
            input_dim=input_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            output_dim=1,
        )
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_cfg = LossConfig(
        mse_weight=args.mse_weight,
        nonneg_weight=args.nonneg_weight,
        sparsity_weight=args.sparsity_weight,
        aod_task_weight=float(args.aod_task_weight),
        t2m_task_weight=0.0,
        ceres_task_weight=0.0,
        rh_task_weight=0.0,
    )
    if args.prediction_mode == "trend_sum":
        logging.info(
            "D5 loss weights | AOD=%.1f T2M=%.1f CERES=%.1f RH=%.1f (aux heads disabled)",
            float(loss_cfg.aod_task_weight),
            float(loss_cfg.t2m_task_weight),
            float(loss_cfg.ceres_task_weight),
            float(loss_cfg.rh_task_weight),
        )

    train_loader = DataLoader(
        FeatureTargetDataset(x_train, y_train, p_train, aux_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        FeatureTargetDataset(x_val, y_val, p_val, aux_val),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        FeatureTargetDataset(x_test, y_test, p_test, aux_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    best_diffusion_state = deepcopy(diffusion_task_model.state_dict()) if diffusion_task_model is not None else None
    no_improve = 0
    train_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_total, train_mse, train_extra = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            loss_cfg=loss_cfg,
            prediction_mode=args.prediction_mode,
            aod_feature_index=aod_feature_index,
            dt_days=args.dt_days,
            rollout_decay=args.rollout_decay,
            max_points_per_forward=args.max_points_per_forward,
            online_physics_coupling=bool(args.online_physics_coupling),
            iterative_aux_forcing=bool(args.iterative_aux_forcing),
            learn_diffusion_coeff=bool(args.learn_diffusion_coeff),
            u10_feature_index=u10_feature_index,
            v10_feature_index=v10_feature_index,
            wind_speed_feature_index=wind_speed_feature_index,
            t2m_feature_index=t2m_feature_index,
            ceres_feature_index=ceres_feature_index,
            rh_feature_index=rh_feature_index,
            physics_grid=physics_grid_t,
            physics_cfg=physics_cfg_t,
            diffusion_task_model=diffusion_task_model,
            diffusion_k_min=float(args.diffusion_k_min),
            diffusion_k_base=float(args.diffusion_k_base),
            diffusion_k_slope=float(args.diffusion_k_slope),
            diffusion_k_cap=float(args.diffusion_k_cap),
        )
        if args.prediction_mode == "trend_sum" and train_extra:
            logging.info(
                "Epoch %03d | train_total=%.6f train_mse=%.6f | phys_trend_mean=%.6e dl_trend_mean=%.6e total_trend_mean=%.6e sparsity=%.6e\n%s\n%s\n%s\n%s",
                epoch,
                train_total,
                train_mse,
                train_extra["phys_trend_mean"],
                train_extra["dl_trend_mean"],
                train_extra["total_trend_mean"],
                train_extra.get("sparsity_penalty", 0.0),
                _format_step_mse(train_extra),
                _format_state_compare(train_extra),
                _format_trend_compare(train_extra),
                _format_process_compare(train_extra),
            )
        else:
            logging.info("Epoch %03d | train_total=%.6f train_mse=%.6f", epoch, train_total, train_mse)

        should_eval = (epoch % max(1, args.eval_interval) == 0) or (epoch == args.epochs)
        if not should_eval:
            continue

        with torch.no_grad():
            val_total, val_mse, val_extra = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                loss_cfg=loss_cfg,
                prediction_mode=args.prediction_mode,
                aod_feature_index=aod_feature_index,
                dt_days=args.dt_days,
                rollout_decay=args.rollout_decay,
                max_points_per_forward=args.max_points_per_forward,
                online_physics_coupling=bool(args.online_physics_coupling),
                iterative_aux_forcing=bool(args.iterative_aux_forcing),
                learn_diffusion_coeff=bool(args.learn_diffusion_coeff),
                u10_feature_index=u10_feature_index,
                v10_feature_index=v10_feature_index,
                wind_speed_feature_index=wind_speed_feature_index,
                t2m_feature_index=t2m_feature_index,
                ceres_feature_index=ceres_feature_index,
                rh_feature_index=rh_feature_index,
                physics_grid=physics_grid_t,
                physics_cfg=physics_cfg_t,
                diffusion_task_model=diffusion_task_model,
                diffusion_k_min=float(args.diffusion_k_min),
                diffusion_k_base=float(args.diffusion_k_base),
                diffusion_k_slope=float(args.diffusion_k_slope),
                diffusion_k_cap=float(args.diffusion_k_cap),
             )
        if args.prediction_mode == "trend_sum" and val_extra:
            logging.info(
                "Epoch %03d | val_total=%.6f val_mse=%.6f | phys_trend_mean=%.6e dl_trend_mean=%.6e total_trend_mean=%.6e sparsity=%.6e\n%s\n%s\n%s\n%s",
                epoch,
                val_total,
                val_mse,
                val_extra["phys_trend_mean"],
                val_extra["dl_trend_mean"],
                val_extra["total_trend_mean"],
                val_extra.get("sparsity_penalty", 0.0),
                _format_step_mse(val_extra),
                _format_state_compare(val_extra),
                _format_trend_compare(val_extra),
                _format_process_compare(val_extra),
            )
        else:
            logging.info("Epoch %03d | val_total=%.6f val_mse=%.6f", epoch, val_total, val_mse)

        if val_total + args.min_delta < best_val:
            best_val = val_total
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            if diffusion_task_model is not None:
                best_diffusion_state = deepcopy(diffusion_task_model.state_dict())
            no_improve = 0
            logging.info("New best validation total loss: %.6f (epoch=%d)", best_val, best_epoch)
        else:
            no_improve += 1
            logging.info("No improvement count: %d/%d", no_improve, args.patience)
            if no_improve >= args.patience:
                logging.info(
                    "Early stopping triggered after %d evaluations without improvement (patience=%d).",
                    no_improve,
                    args.patience,
                )
                break

    train_elapsed_seconds = float(time.perf_counter() - train_start_time)
    train_elapsed_hms = time.strftime("%H:%M:%S", time.gmtime(max(train_elapsed_seconds, 0.0)))
    train_timing_msg = (
        f"Training finished | seed={int(args.seed)} | elapsed={train_elapsed_seconds:.2f}s ({train_elapsed_hms})"
    )
    logging.info(train_timing_msg)
    print(train_timing_msg)

    aod_target_min_eval: float | None = None
    aod_target_range_eval: float | None = None
    aod_denorm_source = "not_requested"
    if stats_paths:
        aod_target_min_eval, aod_target_range_eval, aod_denorm_source = resolve_target_denorm_params(
            target_var=args.target_var,
            stats_paths=stats_paths,
        )
        if aod_target_min_eval is not None and aod_target_range_eval is not None:
            logging.info(
                "Final metrics will be evaluated in physical space using denorm stats: "
                "target=%s min=%.6e range=%.6e source=%s",
                args.target_var,
                aod_target_min_eval,
                aod_target_range_eval,
                aod_denorm_source,
            )
        else:
            logging.warning(
                "No valid denorm stats for target=%s from paths=%s. Metrics remain in normalized space.",
                args.target_var,
                ", ".join(str(p) for p in stats_paths),
            )
    else:
        logging.warning("No data_stats.nc path resolved. Final metrics remain in normalized space.")

    aux_denorm_params_eval: dict[str, tuple[float | None, float | None]] = {}
    aux_denorm_sources_eval: dict[str, str] = {}
    if stats_paths and used_features and args.prediction_mode == "trend_sum" and aux_test is not None:
        aux_feature_keys: dict[str, str] = {
            "t2m": "T2M",
            "ceres": "CERES",
            "rh": "R",
        }
        for aux_key, feat_key in aux_feature_keys.items():
            feat_idx = find_feature_index(used_features, feat_key, required=False)
            if aux_key == "ceres" and feat_idx is None:
                feat_idx = find_feature_index(used_features, "CERES_ALL", required=False)
            if feat_idx is None:
                continue
            var_name = str(used_features[int(feat_idx)])
            v_min, v_range, v_src = resolve_var_denorm_params(var_name=var_name, stats_paths=stats_paths)
            aux_denorm_params_eval[aux_key] = (v_min, v_range)
            aux_denorm_sources_eval[aux_key] = str(v_src)
            if v_min is not None and v_range is not None:
                logging.info(
                    "Aux metrics in physical space will use denorm stats: var=%s min=%.6e range=%.6e source=%s",
                    var_name,
                    float(v_min),
                    float(v_range),
                    str(v_src),
                )
            else:
                logging.warning(
                    "Aux metric var=%s has no valid denorm stats. It remains in normalized space.",
                    var_name,
                )

    model.load_state_dict(best_state)
    if diffusion_task_model is not None and best_diffusion_state is not None:
        diffusion_task_model.load_state_dict(best_diffusion_state)
    final_metrics = evaluate_regression_metrics(
        model,
        test_loader,
        device=device,
        prediction_mode=args.prediction_mode,
        aod_feature_index=aod_feature_index,
        dt_days=args.dt_days,
        rollout_decay=args.rollout_decay,
        max_points_per_forward=args.max_points_per_forward,
        loss_cfg=loss_cfg,
        online_physics_coupling=bool(args.online_physics_coupling),
        iterative_aux_forcing=bool(args.iterative_aux_forcing),
        learn_diffusion_coeff=bool(args.learn_diffusion_coeff),
        u10_feature_index=u10_feature_index,
        v10_feature_index=v10_feature_index,
        wind_speed_feature_index=wind_speed_feature_index,
        t2m_feature_index=t2m_feature_index,
        ceres_feature_index=ceres_feature_index,
        rh_feature_index=rh_feature_index,
        physics_grid=physics_grid_t,
        physics_cfg=physics_cfg_t,
        diffusion_task_model=diffusion_task_model,
        diffusion_k_min=float(args.diffusion_k_min),
        diffusion_k_base=float(args.diffusion_k_base),
        diffusion_k_slope=float(args.diffusion_k_slope),
        diffusion_k_cap=float(args.diffusion_k_cap),
        aod_target_min=aod_target_min_eval,
        aod_target_range=aod_target_range_eval,
        aux_denorm_params=aux_denorm_params_eval,
        aux_denorm_sources=aux_denorm_sources_eval,
    )
    aod_is_physical = aod_target_min_eval is not None and aod_target_range_eval is not None
    final_metrics["metric_space"] = "physical_denormalized" if aod_is_physical else "normalized"
    final_metrics["aod_metric_space"] = str(final_metrics["metric_space"])
    final_metrics["aod_denorm_source"] = aod_denorm_source
    logging.info(
        "Final evaluation metrics (test split, %s) | R2=%.6f RMSE=%.6f MAE=%.6f N_valid=%.0f",
        str(final_metrics["metric_space"]),
        float(final_metrics["r2"]),
        float(final_metrics["rmse"]),
        float(final_metrics["mae"]),
        float(final_metrics["n_valid"]),
    )
    if args.prediction_mode == "trend_sum":
        step_eval_log = _format_step_eval_metrics(final_metrics)
        if step_eval_log:
            logging.info("Final evaluation step metrics (test split)\n%s", step_eval_log)

    if args.prediction_mode == "trend_sum" and aux_test is not None:
        for aux_key, aux_label in (("t2m", "T2M"), ("ceres", "CERES"), ("rh", "RH")):
            r2_key = f"{aux_key}_r2"
            if r2_key not in final_metrics:
                continue
            aux_metric_space = str(final_metrics.get(f"{aux_key}_metric_space", "normalized"))
            aux_metric_source = str(final_metrics.get(f"{aux_key}_denorm_source", "not_found"))
            logging.info(
                "Final evaluation metrics (%s, test split, %s, source=%s, note=%s) | R2=%.6f RMSE=%.6f MAE=%.6f N_valid=%.0f",
                aux_label,
                aux_metric_space,
                aux_metric_source,
                str(final_metrics.get(f"{aux_key}_eval_note", "unknown")),
                float(final_metrics.get(f"{aux_key}_r2", float("nan"))),
                float(final_metrics.get(f"{aux_key}_rmse", float("nan"))),
                float(final_metrics.get(f"{aux_key}_mae", float("nan"))),
                float(final_metrics.get(f"{aux_key}_n_valid", float("nan"))),
            )
            aux_step_metrics = {
                "r2_step": list(final_metrics.get(f"{aux_key}_r2_step", [])),
                "rmse_step": list(final_metrics.get(f"{aux_key}_rmse_step", [])),
                "mae_step": list(final_metrics.get(f"{aux_key}_mae_step", [])),
                "n_valid_step": list(final_metrics.get(f"{aux_key}_n_valid_step", [])),
            }
            aux_step_eval_log = _format_step_eval_metrics(aux_step_metrics)
            if aux_step_eval_log:
                logging.info("Final evaluation step metrics (%s, test split)\n%s", aux_label, aux_step_eval_log)

    if args.eval_nc_out is not None:
        eval_nc_path = args.eval_nc_out
    else:
        eval_nc_path = args.out.with_name(f"{args.out.stem}_eval_outputs.nc")
    pred_eval, target_eval = collect_regression_outputs(
        model,
        test_loader,
        device=device,
        prediction_mode=args.prediction_mode,
        aod_feature_index=aod_feature_index,
        dt_days=args.dt_days,
        rollout_decay=args.rollout_decay,
        max_points_per_forward=args.max_points_per_forward,
        loss_cfg=loss_cfg,
        online_physics_coupling=bool(args.online_physics_coupling),
        iterative_aux_forcing=bool(args.iterative_aux_forcing),
        learn_diffusion_coeff=bool(args.learn_diffusion_coeff),
        u10_feature_index=u10_feature_index,
        v10_feature_index=v10_feature_index,
        wind_speed_feature_index=wind_speed_feature_index,
        t2m_feature_index=t2m_feature_index,
        ceres_feature_index=ceres_feature_index,
        rh_feature_index=rh_feature_index,
        physics_grid=physics_grid_t,
        physics_cfg=physics_cfg_t,
        diffusion_task_model=diffusion_task_model,
        diffusion_k_min=float(args.diffusion_k_min),
        diffusion_k_base=float(args.diffusion_k_base),
        diffusion_k_slope=float(args.diffusion_k_slope),
        diffusion_k_cap=float(args.diffusion_k_cap),
        aod_target_min=aod_target_min_eval,
        aod_target_range=aod_target_range_eval,
    )
    try:
        save_regression_outputs_to_nc(
            out_path=eval_nc_path,
            pred=pred_eval,
            target=target_eval,
            prediction_mode=args.prediction_mode,
            metric_space=str(final_metrics["metric_space"]),
            online_physics_coupling=bool(args.online_physics_coupling),
        )
        logging.info(
            "Saved final evaluation outputs to %s | shape=%s",
            eval_nc_path,
            tuple(pred_eval.shape),
        )
    except Exception as exc:
        logging.exception("Failed to save final evaluation outputs to %s: %s", eval_nc_path, exc)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "diffusion_task_state_dict": (diffusion_task_model.state_dict() if diffusion_task_model is not None else None),
            "input_dim": input_dim,
            "hidden_dims": list(args.hidden_dims),
            "dropout": args.dropout,
            "best_val_total": best_val,
            "best_epoch": best_epoch,
            "feature_keys": list(args.feature_keys),
            "feature_names": used_features,
            "target_var": args.target_var,
            "prediction_mode": args.prediction_mode,
            "physics_var": args.physics_var,
            "horizon": args.horizon,
            "rollout_steps": args.rollout_steps,
            "dt_days": args.dt_days,
            "rollout_decay": args.rollout_decay,
            "online_physics_coupling": bool(args.online_physics_coupling),
            "iterative_aux_forcing": bool(args.iterative_aux_forcing),
            "learn_diffusion_coeff": bool(args.learn_diffusion_coeff),
            "diffusion_mlp_hidden": int(args.diffusion_mlp_hidden),
            "diffusion_mlp_lr": float(args.diffusion_mlp_lr),
            "diffusion_k_min": float(args.diffusion_k_min),
            "diffusion_k_base": float(args.diffusion_k_base),
            "diffusion_k_slope": float(args.diffusion_k_slope),
            "diffusion_k_cap": float(args.diffusion_k_cap),
            "physics_advection_scheme": str(args.physics_advection_scheme),
            "physics_enable_diffusion": bool(args.physics_enable_diffusion),
            "physics_diffusion_k_const": float(args.physics_diffusion_k_const),
            "physics_cos_floor": float(args.physics_cos_floor),
            "max_points_per_forward": args.max_points_per_forward,
            "lr_src": args.lr_src,
            "lr_wet": args.lr_wet,
            "lr_dry": args.lr_dry,
            "sparsity_weight": args.sparsity_weight,
            "aod_task_weight": float(loss_cfg.aod_task_weight),
            "t2m_task_weight": float(loss_cfg.t2m_task_weight),
            "ceres_task_weight": float(loss_cfg.ceres_task_weight),
            "rh_task_weight": float(loss_cfg.rh_task_weight),
            "rad_init_bias": args.rad_init_bias,
            "dry_init_bias": args.dry_init_bias,
            "wet_init_bias": args.wet_init_bias,
            "device": str(device),
            "cuda_device_id": args.cuda_device_id,
            "seed": int(args.seed),
            "train_elapsed_seconds": train_elapsed_seconds,
            "train_elapsed_hms": train_elapsed_hms,
            "train_frac": args.train_frac,
            "source_paths": source_paths,
            "eval_nc_out": str(eval_nc_path),
            "final_test_metrics": final_metrics,
        },
        args.out,
    )
    logging.info("Saved model checkpoint to %s", args.out)
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    run_seeds = _resolve_run_seeds(args)

    if len(run_seeds) == 1 or args.single_seed_run:
        args.seed = int(run_seeds[0])
        return _run_single(args)

    for idx, seed in enumerate(run_seeds, start=1):
        run_args = deepcopy(args)
        run_args.seed = int(seed)
        run_args.num_models = 1
        run_args.single_seed_run = True
        run_args.out = _path_with_seed_suffix(Path(args.out), int(seed))
        if args.eval_nc_out is None:
            run_args.eval_nc_out = run_args.out.with_name(f"{run_args.out.stem}_eval_outputs.nc")
        else:
            run_args.eval_nc_out = _path_with_seed_suffix(Path(args.eval_nc_out), int(seed))
        logging.info(
            "Starting multi-seed run %d/%d | seed=%d | out=%s | eval_nc_out=%s",
            idx,
            len(run_seeds),
            int(seed),
            str(run_args.out),
            str(run_args.eval_nc_out),
        )
        rc = _run_single(run_args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())























































