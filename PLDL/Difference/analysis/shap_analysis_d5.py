from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from Difference.dl import GridPointMLP, MLPConfig
from Difference.settings import GLOBAL_SETTINGS, TRAIN_SETTINGS

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D5 SHAP analysis for pure-DL baseline.")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path)
    parser.add_argument("--data-nc", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--years", type=int, nargs="+", default=list(TRAIN_SETTINGS.get("years", GLOBAL_SETTINGS["years"])))
    parser.add_argument(
        "--feature-keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Feature keys or resolved variable names to load from NetCDF. "
            "If omitted, use ckpt feature_names/feature_keys; otherwise fallback to built-in defaults."
        ),
    )
    parser.add_argument("--require-all-features", action=argparse.BooleanOptionalAction, default=False)
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
        key_s = str(key)
        cands = FEATURE_CANDIDATES.get(key_s.upper())
        if cands is None:
            # Support checkpoints that store resolved variable names directly.
            if key_s in ds.data_vars:
                names.append(key_s)
                continue
            missing.append(f"{key_s} (unknown key)")
            continue
        chosen = next((n for n in cands if n in ds.data_vars), None)
        if chosen is None:
            missing.append(f"{key_s} -> {cands}")
            continue
        names.append(chosen)
    if missing and require_all:
        raise KeyError("Missing required features: " + "; ".join(missing))
    if missing:
        logging.warning("Skipping missing features: %s", "; ".join(missing))
    return names


def load_feature_array_from_nc(
    *,
    data_nc: Path | None,
    data_root: Path | None,
    years: list[int] | None,
    feature_keys: list[str],
    require_all_features: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    if data_nc is not None:
        if not data_nc.exists():
            raise FileNotFoundError(f"NetCDF not found: {data_nc}")
        paths = [data_nc]
    else:
        if data_root is None:
            data_root = Path(GLOBAL_SETTINGS["data_root"])
        filename = str(TRAIN_SETTINGS.get("trend_input_filename", TRAIN_SETTINGS.get("input_filename", "data_normalize.nc")))
        paths = resolve_data_paths(data_root, years, filename)
    ds = xr.open_mfdataset([str(p) for p in paths], combine="by_coords", engine="netcdf4")
    try:
        if "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": TIME_DIM})
        if "number" in ds.coords or "number" in ds:
            ds = ds.drop_vars("number", errors="ignore")
        feat_names = resolve_feature_names(ds, feature_keys, require_all=require_all_features)
        feat_ds = ds[feat_names]
        if "pressure_level" in feat_ds.dims:
            feat_ds = feat_ds.isel(pressure_level=0).drop_vars("pressure_level", errors="ignore")
        if "level" in feat_ds.dims:
            feat_ds = feat_ds.isel(level=0).drop_vars("level", errors="ignore")
        feat_da = feat_ds.to_array("feature").transpose("latitude", "longitude", TIME_DIM, "feature")
        feat = np.transpose(np.asarray(feat_da.values, dtype=np.float32), (2, 0, 1, 3))
    finally:
        ds.close()
    return feat, feat_names, [str(p) for p in paths]


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


def build_model(ckpt: dict, feature_dim: int, device: torch.device) -> torch.nn.Module:
    state = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint missing model_state_dict/state_dict.")
    hidden_dims = tuple(int(v) for v in ckpt.get("hidden_dims", [128, 128]))
    dropout = float(ckpt.get("dropout", 0.1))
    input_dim = int(ckpt.get("input_dim", feature_dim))
    if input_dim != feature_dim:
        raise ValueError(f"Feature dim mismatch between data and ckpt: data={feature_dim}, ckpt={input_dim}")
    model = GridPointMLP(MLPConfig(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, output_dim=1))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        # DDP checkpoints may prefix parameters with "module.".
        stripped = {}
        for k, v in state.items():
            kk = str(k)
            stripped[kk[7:] if kk.startswith("module.") else kk] = v
        model.load_state_dict(stripped, strict=True)
    model = model.to(device)
    model.eval()
    return model


class WrappedD5Model(torch.nn.Module):
    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_model(x)
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


def _align_feature_names(feature_names: list[str], feature_dim: int) -> list[str]:
    names = [str(v) for v in feature_names]
    if len(names) == feature_dim:
        return names
    if len(names) > feature_dim:
        logging.warning(
            "feature_names(%d) > feature_dim(%d). Truncating to match feature dimension.",
            len(names),
            feature_dim,
        )
        return names[:feature_dim]
    logging.warning(
        "feature_names(%d) < feature_dim(%d). Appending generic names for missing dimensions.",
        len(names),
        feature_dim,
    )
    names.extend([f"feature_{i}" for i in range(len(names), feature_dim)])
    return names


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    ckpt_feature_pref = list(ckpt.get("feature_names") or ckpt.get("feature_keys") or [])
    if args.feature_keys is None:
        if ckpt_feature_pref:
            feature_keys_for_load = list(ckpt_feature_pref)
            logging.info("Using feature list from checkpoint for NetCDF loading.")
        else:
            feature_keys_for_load = list(DEFAULT_FEATURE_KEYS)
            logging.info("Checkpoint has no feature list; fallback to default feature keys.")
    else:
        feature_keys_for_load = list(args.feature_keys)

    if args.data_npz is not None:
        x = load_feature_array_from_npz(args.data_npz)
        source_paths = [str(args.data_npz)]
        feature_names = list(ckpt_feature_pref or feature_keys_for_load)
    else:
        x, feature_names, source_paths = load_feature_array_from_nc(
            data_nc=args.data_nc,
            data_root=args.data_root,
            years=args.years,
            feature_keys=feature_keys_for_load,
            require_all_features=bool(args.require_all_features),
        )

    x2d = feature_matrix(x, sample_step=int(args.sample_step))
    feature_names = _align_feature_names(feature_names, int(x2d.shape[1]))
    raw_rows = int(x2d.shape[0])
    if args.max_rows > 0 and x2d.shape[0] > args.max_rows:
        keep = np.random.choice(x2d.shape[0], size=int(args.max_rows), replace=False)
        x2d = x2d[keep]

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

    model = build_model(ckpt, feature_dim=x2d.shape[1], device=device)
    wrapped = WrappedD5Model(model).to(device)
    wrapped.eval()

    try:
        import shap
    except Exception as exc:
        raise RuntimeError("shap is required. Please install with: pip install shap") from exc

    logging.info(
        "Explainer=%s | model=gridpoint_mlp | rows_raw=%d | rows_used=%d | bg=%d | explain=%d | chunk=%d",
        args.explainer,
        raw_rows,
        int(x2d.shape[0]),
        n_bg,
        n_exp,
        int(args.explain_batch_size),
    )
    if args.explainer == "gradient":
        bg_t = torch.from_numpy(bg).to(device=device, dtype=torch.float32)
        explainer = shap.GradientExplainer(wrapped, bg_t)
        chunk = max(1, int(args.explain_batch_size))
        shap_chunks: list[np.ndarray] = []
        for i in range(0, ex.shape[0], chunk):
            ex_t = torch.from_numpy(ex[i : i + chunk]).to(device=device, dtype=torch.float32)
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

        explainer = shap.KernelExplainer(_predict, bg)
        chunk = max(1, int(args.explain_batch_size))
        shap_chunks = []
        for i in range(0, ex.shape[0], chunk):
            shap_raw_chunk = explainer.shap_values(ex[i : i + chunk], nsamples=int(args.nsamples))
            shap_chunks.append(_to_shap_array(shap_raw_chunk))
        shap_values = _stack_shap_chunks(shap_chunks)

    if shap_values.shape[1] != len(feature_names):
        raise ValueError(f"SHAP dim mismatch: shap={shap_values.shape}, feature_names={len(feature_names)}")

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    order = np.argsort(-mean_abs)

    out_dir = args.out_dir / f"{args.ckpt.stem}_model_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "shap_values.npy", shap_values.astype(np.float32))
    np.save(out_dir / "explain_samples.npy", ex.astype(np.float32))

    lines = ["feature,mean_abs_shap,mean_signed_shap,rank"]
    for rank, idx in enumerate(order, start=1):
        lines.append(f"{feature_names[idx]},{mean_abs[idx]:.10e},{mean_signed[idx]:.10e},{rank}")
    (out_dir / "feature_importance.csv").write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "framework": "D5PL",
        "ckpt": str(args.ckpt),
        "model_type": "gridpoint_mlp",
        "prediction_mode": str(ckpt.get("prediction_mode", "unknown")),
        "target": "model_output",
        "feature_names": feature_names,
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
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        import shap

        topk = min(int(args.topk), len(feature_names))
        top_idx = order[:topk][::-1]
        plt.figure(figsize=(10, 0.45 * topk + 2))
        plt.barh([feature_names[i] for i in top_idx], [mean_abs[i] for i in top_idx])
        plt.xlabel("mean(|SHAP|)")
        plt.title("D5 SHAP (pure DL)")
        plt.tight_layout()
        plt.savefig(out_dir / "top_features_bar.png", dpi=180)
        plt.close()

        shap.summary_plot(shap_values, ex, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / "shap_beeswarm.png", dpi=180)
        plt.close()
    except Exception as exc:
        logging.warning("Plot generation skipped: %s", exc)

    logging.info("Saved SHAP outputs to: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
