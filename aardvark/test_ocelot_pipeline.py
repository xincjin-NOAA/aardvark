"""
End-to-end smoke test for the ocelot3 data adapter (ocelot_loader.py +
ocelot_models.py): generates a tiny synthetic URMA-shaped Parquet dataset on
disk (matching ocelot3's real Hive-partitioned layout and obs_config_urma
schema), then runs ocelot3's *real*, unmodified ParquetDataManager /
extract_features_from_df through OcelotWeatherDataset and one forward+backward
pass of OcelotAardvarkModel, checking shapes and a finite loss.

This does NOT validate against real URMA data or real model behavior/accuracy
-- it only validates that the plumbing (schema assumptions, tensor shapes,
convDeepSet/ViT wiring) doesn't crash and produces sane shapes. Run this before
pointing train_ocelot.py at real data.

Run this in the conda env that has torch/timm/pandas/pyarrow/scikit-learn
(ocelot3 itself is no longer needed -- see ocelot_vendor/):
    python test_ocelot_pipeline.py
"""
import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from ocelot_loader import OcelotWeatherDataset, collate_single
from ocelot_models import OcelotAardvarkModel, masked_mse_loss

BIN_DATE = "2025-02-08"
BIN_HOUR = "13"
DELTA_TIME = 1
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 240.0, 260.0, 30.0, 45.0  # small synthetic sub-domain
N_POINT_OBS = 300
N_GRID = 800


def _synthetic_point_df(feature_keys, metadata_keys, feature_stats, qc_filters, n_rows, rng):
    lat = rng.uniform(LAT_MIN + 1, LAT_MAX - 1, size=n_rows)
    lon = rng.uniform(LON_MIN + 1, LON_MAX - 1, size=n_rows)
    ts = pd.Timestamp(f"{BIN_DATE} {BIN_HOUR}:00:00").timestamp()
    obs_time = ts + rng.uniform(-300, 300, size=n_rows)

    data = {"latitude": lat, "longitude": lon, "obs_time": obs_time, "analysis_use_flag": np.ones(n_rows)}

    for key in feature_keys:
        if key in ("pressure", "airPressure", "pressureMeanSeaLevel_bufr"):
            data[key] = rng.uniform(950.0, 1030.0, size=n_rows)  # raw hPa; loader log-transforms it
        else:
            qc = (qc_filters or {}).get(key, {})
            if "range" in qc:
                lo, hi = qc["range"]
                margin = 0.1 * (hi - lo)
                data[key] = rng.uniform(lo + margin, hi - margin, size=n_rows)
            elif key in feature_stats:
                mean, std = feature_stats[key]
                data[key] = mean + std * rng.normal(size=n_rows) * 0.1
            else:
                data[key] = rng.normal(size=n_rows)

    for key in metadata_keys or []:
        if key in feature_stats:
            mean, std = feature_stats[key]
            data[key] = np.abs(mean + std * rng.normal(size=n_rows) * 0.1)
        else:
            data[key] = rng.uniform(0, 100, size=n_rows)

    return pd.DataFrame(data)


def _synthetic_state_df(feature_keys, feature_stats, n_rows, rng, perturb=False):
    lat = rng.uniform(LAT_MIN + 0.5, LAT_MAX - 0.5, size=n_rows)
    lon = rng.uniform(LON_MIN + 0.5, LON_MAX - 0.5, size=n_rows)
    ts = pd.Timestamp(f"{BIN_DATE} {BIN_HOUR}:00:00").timestamp()
    obs_time = np.full(n_rows, ts)
    data = {"latitude": lat, "longitude": lon, "obs_time": obs_time}
    for key in feature_keys:
        mean, std = feature_stats.get(key, [0.0, 1.0])
        base = mean + 0.1 * std * rng.normal(size=n_rows)
        if perturb:
            base = base + 0.3 * std  # nonzero increment for anal vs ges
        data[key] = base
    return pd.DataFrame(data)


def _write_partition(data_dir, file_base, year, date_part, cycle, df):
    part_dir = os.path.join(data_dir, f"{file_base}_{year}.parquet", f"date={date_part}", f"cycle={cycle}")
    os.makedirs(part_dir, exist_ok=True)
    df.to_parquet(os.path.join(part_dir, "part-0.parquet"))


def build_synthetic_dataset(tmp_dir, seed=0):
    from ocelot_vendor.observation_config import load_observation_config

    obs_config, feature_stats, fill_values, _, increment_stats = load_observation_config(
        exp_type="regional_da", config_name="urma"
    )
    rng = np.random.default_rng(seed)
    year = BIN_DATE.split("-")[0]

    for obs_type, insts in obs_config.items():
        if obs_type == "state":
            continue
        for inst_name, cfg in insts.items():
            file_base = cfg.get("zarr_name", inst_name)
            df = _synthetic_point_df(
                cfg["features"],
                cfg.get("metadata"),
                feature_stats.get(inst_name, {}),
                cfg.get("qc_filters"),
                N_POINT_OBS,
                rng,
            )
            _write_partition(tmp_dir, file_base, year, BIN_DATE, BIN_HOUR, df)

    ges_cfg = obs_config["state"]["ges"]
    ges_df = _synthetic_state_df(ges_cfg["features"], feature_stats.get("ges", {}), N_GRID, rng, perturb=False)
    _write_partition(tmp_dir, ges_cfg.get("zarr_name", "ges"), year, BIN_DATE, BIN_HOUR, ges_df)

    anal_zarr_name = ges_cfg.get("anal_zarr_name", "anal")
    anal_df = _synthetic_state_df(ges_cfg["features"], feature_stats.get("ges", {}), N_GRID, rng, perturb=True)
    _write_partition(tmp_dir, anal_zarr_name, year, BIN_DATE, BIN_HOUR, anal_df)

    static_dir = os.path.join(tmp_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    terrain = rng.uniform(0, 2000, size=N_GRID).astype(np.float32)
    slmask = rng.integers(0, 2, size=N_GRID).astype(np.float32)
    np.save(os.path.join(static_dir, "urma2p5_terrain.npy"), terrain)
    np.savez(os.path.join(static_dir, "urma2p5_slmask_nolakes.npz"), slmask=slmask)

    return static_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="ocelot_aardvark_smoke_")
    try:
        static_dir = build_synthetic_dataset(tmp_dir)

        dataset = OcelotWeatherDataset(
            data_path=tmp_dir,
            static_data_dir=static_dir,
            start_date=BIN_DATE,
            end_date=BIN_DATE,
            delta_time=DELTA_TIME,
            obs_config_name="urma",
            exp_type="regional_da",
            target_sample_size=200,
        )
        # generate_binned_timestamp_list produces one bin per hour of BIN_DATE
        # (delta_time=1); synthetic Parquet data was only written for BIN_HOUR,
        # so every other hourly bin is legitimately empty/None -- find the one
        # bin we actually populated rather than assuming index 0 or len==1.
        target_bin_name = f"day_half={BIN_DATE}_{BIN_HOUR}"
        assert target_bin_name in dataset.binned_samples, dataset.binned_samples
        idx = dataset.binned_samples.index(target_bin_name)

        task = dataset[idx]
        assert task is not None, "dataset returned None for the synthetic bin -- check synthetic schema"
        print("Instrument channel dims:", dataset.instrument_channel_dims())
        print("y_target shape:", task["y_target"].shape)
        print("x_target shape:", task["x_target"].shape)

        subset = torch.utils.data.Subset(dataset, [idx])
        loader = torch.utils.data.DataLoader(subset, batch_size=1, collate_fn=collate_single)
        batch = next(iter(loader))
        assert batch is not None

        device = torch.device(args.device)
        model = OcelotAardvarkModel(
            instrument_channels=dataset.instrument_channel_dims(),
            target_dim=dataset.target_dim,
            lon_min=LON_MIN,
            lon_max=LON_MAX,
            lat_min=LAT_MIN,
            lat_max=LAT_MAX,
            grid_nlon=16,
            grid_nlat=16,
            patch_size=4,
            vit_embed_dim=32,
            vit_depth=1,
            vit_num_heads=2,
            int_channels=8,
            mlp_h_channels=16,
            mlp_h_layers=1,
            device=device,
        ).to(device)

        def to_device(t):
            out = {}
            for k, v in t.items():
                if torch.is_tensor(v):
                    out[k] = v.to(device)
                elif isinstance(v, list) and len(v) == 2:
                    out[k] = [v[0].to(device), v[1].to(device)]
                else:
                    out[k] = v
            return out

        batch = to_device(batch)
        y_hat = model(batch)
        print("y_hat shape:", y_hat.shape)
        assert y_hat.shape == batch["y_target"].shape, (y_hat.shape, batch["y_target"].shape)
        assert torch.isfinite(y_hat).all(), "model produced non-finite output"

        loss = masked_mse_loss(y_hat, batch["y_target"], batch["y_target_valid_mask"])
        print("loss:", loss.item())
        assert torch.isfinite(loss)

        loss.backward()
        n_grad = sum(1 for p in model.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        n_total = sum(1 for p in model.parameters())
        print(f"params with finite grad: {n_grad}/{n_total}")

        # Full-grid chunked decode path (used at eval/inference instead of the
        # training-time random subsample).
        full_lon = torch.rand(50) * (LON_MAX - LON_MIN) / 360 + LON_MIN / 360
        full_lat = torch.rand(50) * (LAT_MAX - LAT_MIN) / 360 + LAT_MIN / 360
        full_static = torch.rand(50, 2)
        full_pred = model.predict_full_grid(batch, full_lon, full_lat, full_static, chunk_size=17)
        assert full_pred.shape == (50, dataset.target_dim), full_pred.shape
        print("predict_full_grid shape:", full_pred.shape)

        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
