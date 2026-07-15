from __future__ import annotations

from pathlib import Path


GLOBAL_SETTINGS = {
    "data_root": Path(r"/mnt/sda/lg/data/"),
    "years": [2020, 2021],
    "input_filename": "data_normalize.nc",
    "output_physics_filename": "data_with_physics.nc",
}


TRAIN_SETTINGS = {
    # Data source priority in train script:
    # CLI(data_npz/data_nc/data_root) > settings(train_data_npz/train_data_nc/train_data_root) > GLOBAL_SETTINGS["data_root"]
    "train_data_npz": None,
    "train_data_nc": None,
    "train_data_root": None,
    "years": [2020],
    "input_filename": "data_normalize.nc",
    # Old train_mlp_baseline.py compatibility keys
    "trend_input_filename": "data_normalize.nc",
    "target_var": "AODANA",
    "physics_var": None,
    "prediction_mode": "trend_sum",
    "online_physics_coupling": False,
    "rollout_steps": 1,
    "dt_days": 1.0,
    "rollout_decay": 1.0,
    "physics_align": "input",
    "enable_physics_mainline": True,
    "horizon": 1,
    "train_frac": 0.8,
    "epochs": 100,
    "eval_interval": 5,
    "patience": 10,
    "min_delta": 1e-6,
    "batch_size": 1024,
    "max_points_per_forward": 2000000,
    "lr": 1e-3,
    "lr_src": 1e-3,
    "lr_wet": 1e-3,
    "lr_dry": 1e-4,
    "dropout": 0.1,
    "hidden_dims": [128, 128],
    "mse_weight": 1.0,
    "nonneg_weight": 0.0,
    "sparsity_weight": 1e-4,
    "physics_forward_mode": "pure_dl",  # choices: pure_dl, coupled
    "physics_loss_normalize": True,
    "rad_init_bias": -1.0,
    "dry_init_bias": -3.0,
    "wet_init_bias": -3.0,
    "batch_size_trend_sum": 1,
    "require_all_features": False,
    "device": "cuda",
    "cuda_device_id": 0,
    "out_ckpt": Path("Difference/outputs/mlp_baseline.pt"),
    "num_models": 1,
    "seed_list": [42],
}


PHYSICS_SETTINGS = {
    # Data source priority in physics builder:
    # CLI(data_nc/data_root) > settings(physics_data_nc/physics_data_root) > GLOBAL_SETTINGS["data_root"]
    "physics_data_nc": None,
    "physics_data_root": None,
    "years": [2020],
    "aod_var": "AODANA",
    "u_var": "u10",
    "v_var": "v10",
    "scheme": "semi_lagrangian",
    "dt_days": 1.0,
    "cos_floor": 0.05,
    "device": "cpu",
}
