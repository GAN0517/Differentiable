from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# Allow running as both:
# 1) python Difference/data/build_physics_trend.py
# 2) python -m Difference.data.build_physics_trend
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from Difference.physics.core import PhysicsConfig, physics_tendency
from Difference.physics.grid import SphericalGrid
from Difference.settings import GLOBAL_SETTINGS, PHYSICS_SETTINGS

TIME_DIM = "time"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build physics trend fields (e.g., tendency_total) from normalized dataset."
    )
    parser.add_argument("--data-nc", type=Path, help="Single NetCDF file, e.g. data_normalize.nc")
    parser.add_argument("--data-root", type=Path, help="Root with data_normalize.nc or <year>/data_normalize.nc")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(PHYSICS_SETTINGS.get("years", GLOBAL_SETTINGS["years"])),
        help="Years when --data-root is used.",
    )

    parser.add_argument("--input-filename", type=str, default=GLOBAL_SETTINGS["input_filename"])
    parser.add_argument("--output-filename", type=str, default=GLOBAL_SETTINGS["output_physics_filename"])
    parser.add_argument("--inplace", action="store_true", help="Write physics variables back to input file path.")

    parser.add_argument("--aod-var", type=str, default=PHYSICS_SETTINGS["aod_var"])
    parser.add_argument("--u-var", type=str, default=PHYSICS_SETTINGS["u_var"])
    parser.add_argument("--v-var", type=str, default=PHYSICS_SETTINGS["v_var"])

    parser.add_argument("--scheme", choices=("semi_lagrangian", "central"), default=PHYSICS_SETTINGS["scheme"])
    parser.add_argument("--dt-days", type=float, default=PHYSICS_SETTINGS["dt_days"])
    parser.add_argument(
        "--enable-diffusion",
        action=argparse.BooleanOptionalAction,
        default=bool(PHYSICS_SETTINGS.get("enable_diffusion", False)),
        help="Enable diffusion term div(K*grad(A)) in physics tendency.",
    )
    parser.add_argument(
        "--diffusion-k-const",
        type=float,
        default=float(PHYSICS_SETTINGS.get("diffusion_k_const", 0.0)),
        help="Constant diffusion coefficient K when diffusion is enabled.",
    )
    parser.add_argument("--cos-floor", type=float, default=PHYSICS_SETTINGS["cos_floor"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default=PHYSICS_SETTINGS["device"])
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args()
    if args.data_nc is None and PHYSICS_SETTINGS.get("physics_data_nc") is not None:
        args.data_nc = Path(PHYSICS_SETTINGS["physics_data_nc"])
    if args.data_root is None:
        if PHYSICS_SETTINGS.get("physics_data_root") is not None:
            args.data_root = Path(PHYSICS_SETTINGS["physics_data_root"])
        else:
            args.data_root = Path(GLOBAL_SETTINGS["data_root"])
    return args


def resolve_data_paths(base: Path, years: list[int] | None, filename: str) -> list[Path]:
    if not years:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        return [path]
    paths: list[Path] = []
    for year in years:
        path = base / f"D{int(year)}" / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}. "
                f"D2 expects year folders as D<year> (e.g. D{int(year)})."
            )
        paths.append(path)
    return paths


def load_dataset(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "valid_time" in ds.coords and TIME_DIM not in ds.coords:
        ds = ds.rename({"valid_time": TIME_DIM})
    if "number" in ds.coords or "number" in ds:
        ds = ds.drop_vars("number", errors="ignore")
    if "pressure_level" in ds.dims:
        ds = ds.isel(pressure_level=0).drop("pressure_level")
    if "level" in ds.dims:
        ds = ds.isel(level=0).drop("level")
    ds = ds.sortby(["latitude", "longitude", TIME_DIM])
    return ds


def compute_physics_fields(
    ds: xr.Dataset,
    *,
    aod_var: str,
    u_var: str,
    v_var: str,
    cfg: PhysicsConfig,
    cos_floor: float,
    device: torch.device,
) -> xr.Dataset:
    for name in (aod_var, u_var, v_var):
        if name not in ds.data_vars:
            raise KeyError(f"Required variable missing in dataset: {name}")

    lat = ds["latitude"].values
    lon = ds["longitude"].values
    t_vals = ds[TIME_DIM].values
    lat_n = lat.shape[0]
    lon_n = lon.shape[0]
    t_n = t_vals.shape[0]

    grid = SphericalGrid(
        lat_deg=torch.from_numpy(lat.astype(np.float32)).to(device),
        lon_deg=torch.from_numpy(lon.astype(np.float32)).to(device),
        cos_floor=cos_floor,
    )

    aod = ds[aod_var].transpose("latitude", "longitude", TIME_DIM).values.astype(np.float32)
    u = ds[u_var].transpose("latitude", "longitude", TIME_DIM).values.astype(np.float32)
    v = ds[v_var].transpose("latitude", "longitude", TIME_DIM).values.astype(np.float32)

    tendency_total = np.full((lat_n, lon_n, t_n), np.nan, dtype=np.float32)
    tendency_adv = np.full((lat_n, lon_n, t_n), np.nan, dtype=np.float32)
    tendency_compression = np.full((lat_n, lon_n, t_n), np.nan, dtype=np.float32)
    tendency_diffusion = np.full((lat_n, lon_n, t_n), np.nan, dtype=np.float32)
    divergence = np.full((lat_n, lon_n, t_n), np.nan, dtype=np.float32)

    for t in range(t_n):
        aod_t = aod[:, :, t]
        u_t = u[:, :, t]
        v_t = v[:, :, t]
        valid = np.isfinite(aod_t) & np.isfinite(u_t) & np.isfinite(v_t)

        aod_safe = np.nan_to_num(aod_t, nan=0.0)
        u_safe = np.nan_to_num(u_t, nan=0.0)
        v_safe = np.nan_to_num(v_t, nan=0.0)

        aod_tensor = torch.from_numpy(aod_safe).to(device).unsqueeze(0)
        u_tensor = torch.from_numpy(u_safe).to(device).unsqueeze(0)
        v_tensor = torch.from_numpy(v_safe).to(device).unsqueeze(0)

        tend = physics_tendency(aod_tensor, u_tensor, v_tensor, grid, cfg=cfg)

        total_np = tend["tendency_total"].squeeze(0).detach().cpu().numpy()
        adv_np = tend["tendency_adv"].squeeze(0).detach().cpu().numpy()
        comp_np = tend["tendency_compression"].squeeze(0).detach().cpu().numpy()
        diff_np = tend["tendency_diffusion"].squeeze(0).detach().cpu().numpy()
        div_np = tend["divergence"].squeeze(0).detach().cpu().numpy()

        total_np[~valid] = np.nan
        adv_np[~valid] = np.nan
        comp_np[~valid] = np.nan
        diff_np[~valid] = np.nan
        div_np[~valid] = np.nan

        tendency_total[:, :, t] = total_np
        tendency_adv[:, :, t] = adv_np
        tendency_compression[:, :, t] = comp_np
        tendency_diffusion[:, :, t] = diff_np
        divergence[:, :, t] = div_np

        if t % 30 == 0 or t == t_n - 1:
            logging.info("Computed physics tendency for time index %d/%d", t + 1, t_n)

    return xr.Dataset(
        data_vars={
            "tendency_total": (("latitude", "longitude", TIME_DIM), tendency_total),
            "tendency_adv": (("latitude", "longitude", TIME_DIM), tendency_adv),
            "tendency_compression": (("latitude", "longitude", TIME_DIM), tendency_compression),
            "tendency_diffusion": (("latitude", "longitude", TIME_DIM), tendency_diffusion),
            "wind_divergence_phys": (("latitude", "longitude", TIME_DIM), divergence),
        },
        coords={
            "latitude": ds["latitude"].values,
            "longitude": ds["longitude"].values,
            TIME_DIM: t_vals,
        },
        attrs={
            "physics_scheme": cfg.advection_scheme,
            "dt_days": cfg.dt_days,
            "enable_diffusion": bool(cfg.enable_diffusion),
            "diffusion_k_const": float(cfg.diffusion_k_const),
            "source_aod_var": aod_var,
            "source_u_var": u_var,
            "source_v_var": v_var,
        },
    )


def process_one_path(path: Path, args: argparse.Namespace) -> None:
    logging.info("Processing %s", path)
    ds = load_dataset(path)
    try:
        cfg = PhysicsConfig(
            dt_days=float(args.dt_days),
            advection_scheme=str(args.scheme),
            enable_diffusion=bool(args.enable_diffusion),
            diffusion_k_const=float(args.diffusion_k_const),
        )
        device = torch.device(args.device)
        phys_ds = compute_physics_fields(
            ds,
            aod_var=args.aod_var,
            u_var=args.u_var,
            v_var=args.v_var,
            cfg=cfg,
            cos_floor=args.cos_floor,
            device=device,
        )

        if args.inplace:
            # Avoid overwriting a file that is still opened by xarray/netCDF backend.
            ds_mem = ds.load()
            ds.close()
            merged = xr.merge([ds_mem, phys_ds], join="left", compat="override")
            tmp_path = path.with_name(path.name + ".tmp")
            logging.info("Writing temp file %s", tmp_path)
            merged.to_netcdf(tmp_path)
            tmp_path.replace(path)
            logging.info("Replaced %s in place", path)
        else:
            merged = xr.merge([ds, phys_ds], join="left", compat="override")
            out_path = path.with_name(args.output_filename)
            logging.info("Writing %s", out_path)
            merged.to_netcdf(out_path)
    finally:
        try:
            ds.close()
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if args.data_nc is not None:
        paths = [args.data_nc]
    elif args.data_root is not None:
        paths = resolve_data_paths(args.data_root, args.years, args.input_filename)
    else:
        raise ValueError("No valid data source found. Set data path in settings.py or provide CLI path.")

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")
        process_one_path(p, args)

    logging.info("Physics trend generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


