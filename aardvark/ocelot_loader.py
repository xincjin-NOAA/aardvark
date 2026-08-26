"""
Data loader that reads ocelot3's URMA Parquet observation pipeline directly and
produces Aardvark-style per-instrument tensors, so Aardvark's convDeepSet-based
encoder/decoder (see ocelot_models.py) can run standalone against real URMA data
instead of Aardvark's own loader.py, whose data_dir is a placeholder path
("path_to_data/") that was never wired up to a real dataset.

Reuses ocelot3's raw loading/QC/normalization code (ParquetDataManager,
extract_features_from_df via load_observation_config) via the vendored copy in
ocelot_vendor/ (see ocelot_vendor/__init__.py for what was vendored and why --
short version: Aardvark's own conda env is Python 3.8, and ocelot3's real
dataset_timeseries.py uses Python 3.10+ syntax, so this adapter carries its own
fixed copy instead of depending on a live, version-matched ocelot3 checkout).

Design notes (why this isn't a 1:1 port of Aardvark's loader.py):
  - ocelot3 has no dense-grid loading step; every instrument, including the URMA
    background/analysis field ("state"), is a flat table of (lat, lon, value)
    rows. So every instrument here is fed through convDeepSet in "OffToOn" mode
    (Aardvark's own AMSU/HIRS/GRIDSAT "OnToOn" pre-gridded-swath path is unused,
    since ocelot3 has no equivalent pre-gridded modality).
  - ocelot3's instrument list is config-driven (ocelot/obs_config_urma.py), not
    Aardvark's hardcoded encoder_hadisd/encoder_amsua/etc. methods -- so the task
    dict keys here are named after ocelot3's instruments (e.g. "conventional_diag_t")
    rather than Aardvark's fixed modality names.
  - The DA target is ocelot3's analysis increment (anal_norm - ges_norm), read
    with the same terrain/land-sea-mask static context and increment-normalization
    logic as ocelot3's own DARegionalGraphDataset, so a model trained here is
    predicting the same quantity ocelot3's own model predicts.
  - The analysis grid is ~3.7M cells/bin -- too many query points to decode in one
    forward pass. `target_sample_size` subsamples per __getitem__ for training;
    for full-grid inference, decode in chunks (see OcelotOnToOffDecoder.forward
    called in a loop) instead of raising target_sample_size to "all".
"""
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class OcelotWeatherDataset(Dataset):
    """
    One sample = one URMA time bin (bin width = `delta_time` hours).

    Produces a `task` dict:
      task["{obs_type}_{inst_name}_x_current"] = [lon (1,N), lat (1,N)] or None
      task["{obs_type}_{inst_name}_current"]    = (1,C,N) or None, NaN where
          invalid/missing (matches Aardvark's convDeepSet "no observation" convention)
      task["x_target"]  = (1,2,M) lon/lat of M sampled analysis-grid query points
      task["y_target"]  = (1,M,target_dim) DA increment (or full analysis, see
          target_is_anal) at the query points
      task["y_target_valid_mask"] = (1,M,target_dim) bool
      task["static_at_target"] = (1,2,M) terrain/slmask at the query points
      task["ges_at_target"] = (1,M,target_dim) background value at query points,
          for reconstructing the full analysis at inference: anal = ges + y_hat
      task["lt"] = (1,1) zeros -- DA is single-time, not multi-horizon, but
          ViT.forward requires a lead-time tensor.
    """

    STATE_KEY = "state_ges"

    def __init__(
        self,
        data_path: str,
        static_data_dir: str,
        start_date: str,
        end_date: str,
        delta_time: int = 1,
        obs_config_name: str = "urma",
        exp_type: str = "regional_da",
        target_is_anal: bool = False,
        normalize_increment: bool = False,
        target_sample_size: Optional[int] = 20000,
        seed: int = 0,
    ):
        from ocelot_vendor.observation_config import load_observation_config
        from ocelot_vendor.dataset_timeseries import (
            ParquetDataManager,
            generate_binned_timestamp_list,
        )
        import pandas as pd

        (
            self.observation_config,
            self.feature_stats,
            self.fill_values,
            self.instrument_weights,
            self.increment_stats,
        ) = load_observation_config(exp_type=exp_type, config_name=obs_config_name)

        if "state" not in self.observation_config or "ges" not in self.observation_config["state"]:
            raise ValueError(
                f"exp_type={exp_type!r} obs_config={obs_config_name!r} has no 'state/ges' "
                "instrument configured -- the DA target (analysis increment) needs it. "
                "Use exp_type='regional_da' with the 'urma' config."
            )
        self.ges_cfg = self.observation_config["state"]["ges"]
        # Every non-"state" obs_type/instrument is a plain point-obs encoder input.
        self.point_obs_config = {
            obs_type: insts for obs_type, insts in self.observation_config.items() if obs_type != "state"
        }

        self.data_path = data_path
        self.delta_time = delta_time
        self.target_is_anal = target_is_anal
        self.normalize_increment = normalize_increment
        self.target_sample_size = target_sample_size
        self._rng = np.random.default_rng(seed)

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        self.binned_samples = generate_binned_timestamp_list(
            start.strftime("%m/%d"), end.strftime("%m/%d"), start.year, delta_time
        )

        self.loader = ParquetDataManager(
            data_dir=data_path,
            observation_config=self.observation_config,
            feature_stats=self.feature_stats,
            fill_values=self.fill_values,
            delta_time=delta_time,
        )
        # Mirrors ocelot3's DARegionalGraphDataset.setup(): "anal" is not a real
        # top-level obs_config key, it's ges's config read from a different
        # zarr/parquet basename, sharing ges's normalization stats so the
        # anal_norm - ges_norm subtraction is valid.
        anal_zarr_name = self.ges_cfg.get("anal_zarr_name", "anal")
        anal_obs_config = {"state": {"anal": {**self.ges_cfg, "zarr_name": anal_zarr_name}}}
        anal_feature_stats = {**self.feature_stats, "anal": self.feature_stats.get("ges", {})}
        self.anal_loader = ParquetDataManager(
            data_dir=data_path,
            observation_config=anal_obs_config,
            feature_stats=anal_feature_stats,
            fill_values=self.fill_values,
            delta_time=delta_time,
        )

        terrain = np.load(os.path.join(static_data_dir, "urma2p5_terrain.npy")).astype(np.float32).flatten()
        slmask = np.load(os.path.join(static_data_dir, "urma2p5_slmask_nolakes.npz"))["slmask"].flatten()
        self.terrain = torch.from_numpy((terrain - terrain.mean()) / terrain.std())
        self.slmask = torch.from_numpy(slmask.astype(np.float32))

        if self.target_is_anal and self.normalize_increment:
            raise ValueError("target_is_anal and normalize_increment are mutually exclusive.")
        if self.normalize_increment:
            feature_keys = self.ges_cfg["features"]
            missing = [k for k in feature_keys if k not in self.increment_stats]
            if missing:
                raise ValueError(
                    f"normalize_increment=True but increment_stats is missing entries for: "
                    f"{missing}. Compute them with ocelot/compute_increment_stats.py."
                )
            mean = torch.tensor([self.increment_stats[k][0] for k in feature_keys], dtype=torch.float32)
            std = torch.tensor([self.increment_stats[k][1] for k in feature_keys], dtype=torch.float32)
            self.increment_mean = mean
            self.increment_std = torch.where(std > 0, std, torch.ones_like(std))

    def __len__(self) -> int:
        return len(self.binned_samples)

    def instrument_keys(self) -> List[str]:
        """Ordered list of point-obs instrument keys, e.g. 'conventional_diag_t', plus 'state_ges'."""
        keys = [
            f"{obs_type}_{inst_name}"
            for obs_type, insts in self.point_obs_config.items()
            for inst_name in insts.keys()
        ]
        keys.append(self.STATE_KEY)
        return keys

    def instrument_channel_dims(self) -> Dict[str, int]:
        """{instrument_key: n_channels fed to convDeepSet}. 'state_ges' gets +2 for terrain/slmask."""
        dims = {}
        for obs_type, insts in self.point_obs_config.items():
            for inst_name, cfg in insts.items():
                dims[f"{obs_type}_{inst_name}"] = cfg["target_dim"]
        dims[self.STATE_KEY] = self.ges_cfg["target_dim"] + 2
        return dims

    @property
    def target_dim(self) -> int:
        return self.ges_cfg["target_dim"]

    @staticmethod
    def _nan_where_invalid(features_norm: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        out = features_norm.clone()
        out[~valid_mask.bool()] = float("nan")
        return out

    def _instrument_tensors(self, inst_data: Dict, extra_channels: Optional[torch.Tensor] = None):
        """-> (x_lon (1,N), x_lat (1,N), values (1,C,N)), lon/lat scaled by /360 (Aardvark's convention)."""
        lon = torch.as_tensor(np.asarray(inst_data["lon_deg"], dtype=np.float32)) / 360.0
        lat = torch.as_tensor(np.asarray(inst_data["lat_deg"], dtype=np.float32)) / 360.0
        values = self._nan_where_invalid(inst_data["features_norm"], inst_data["features_valid_mask"])
        if extra_channels is not None:
            values = torch.cat([values, extra_channels], dim=1)
        return lon.unsqueeze(0), lat.unsqueeze(0), values.t().unsqueeze(0)

    @staticmethod
    def _nonempty(inst_data: Optional[Dict]) -> bool:
        if inst_data is None:
            return False
        fn = inst_data.get("features_norm")
        return (
            fn is not None
            and (not torch.is_tensor(fn) or fn.numel() > 0)
            and len(inst_data.get("lat_deg", [])) > 0
        )

    def __getitem__(self, idx: int) -> Optional[Dict]:
        bin_name = self.binned_samples[idx]
        bin_data = self.loader.get_data_for_bin(bin_name)
        anal_bin = self.anal_loader.get_data_for_bin(bin_name)
        if bin_data is None or anal_bin is None:
            return None
        ges_data = bin_data.get("state", {}).get("ges")
        anal_data = anal_bin.get("state", {}).get("anal")
        if not self._nonempty(ges_data) or not self._nonempty(anal_data):
            return None

        task: Dict = {}
        for obs_type, insts in self.point_obs_config.items():
            for inst_name in insts.keys():
                key = f"{obs_type}_{inst_name}"
                inst_data = bin_data.get(obs_type, {}).get(inst_name)
                if not self._nonempty(inst_data):
                    task[f"{key}_x_current"] = None
                    task[f"{key}_current"] = None
                    continue
                x_lon, x_lat, values = self._instrument_tensors(inst_data)
                task[f"{key}_x_current"] = [x_lon, x_lat]
                task[f"{key}_current"] = values

        n_grid = ges_data["features_norm"].shape[0]
        if anal_data["features_norm"].shape[0] != n_grid:
            return None
        if self.terrain.shape[0] < n_grid or self.slmask.shape[0] < n_grid:
            raise ValueError(
                f"Static terrain/slmask grid ({self.terrain.shape[0]}) is smaller than the "
                f"state row count ({n_grid}) for bin {bin_name} -- static_data_dir doesn't "
                f"match this URMA grid."
            )
        static_ctx_full = torch.stack([self.terrain[:n_grid], self.slmask[:n_grid]], dim=1)
        x_lon, x_lat, values = self._instrument_tensors(ges_data, extra_channels=static_ctx_full)
        task[f"{self.STATE_KEY}_x_current"] = [x_lon, x_lat]
        task[f"{self.STATE_KEY}_current"] = values

        ges_features = ges_data["features_norm"].float()
        anal_features = anal_data["features_norm"].float()
        anal_valid_mask = anal_data["features_valid_mask"].bool()

        if self.target_is_anal:
            target = anal_features
        else:
            target = anal_features - ges_features
            if self.normalize_increment:
                target = (target - self.increment_mean) / self.increment_std

        target_lon = torch.as_tensor(np.asarray(anal_data["lon_deg"], dtype=np.float32)) / 360.0
        target_lat = torch.as_tensor(np.asarray(anal_data["lat_deg"], dtype=np.float32)) / 360.0

        if self.target_sample_size is not None and target.shape[0] > self.target_sample_size:
            sel = self._rng.choice(target.shape[0], size=self.target_sample_size, replace=False)
            sel_t = torch.from_numpy(sel).long()
            target = target[sel_t]
            anal_valid_mask = anal_valid_mask[sel_t]
            target_lon = target_lon[sel_t]
            target_lat = target_lat[sel_t]
            ges_at_target = ges_features[sel_t]
            static_at_target = static_ctx_full[sel_t]
        else:
            ges_at_target = ges_features
            static_at_target = static_ctx_full

        task["x_target"] = torch.stack([target_lon, target_lat], dim=0).unsqueeze(0)
        task["y_target"] = target.unsqueeze(0)
        task["y_target_valid_mask"] = anal_valid_mask.unsqueeze(0)
        task["static_at_target"] = static_at_target.t().unsqueeze(0)
        task["ges_at_target"] = ges_at_target.unsqueeze(0)
        task["lt"] = torch.zeros(1, 1)
        task["bin_name"] = bin_name
        return task


def collate_single(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Use with DataLoader(batch_size=1, collate_fn=collate_single).

    Instruments have a different obs count N per bin, and convDeepSet's OffToOn
    path takes N as a plain tensor dimension (no padding/masking support across
    batch elements), so unlike ocelot3's PyG-batched HeteroData graphs, samples
    here can't be stacked across bins. batch_size stays 1; use gradient
    accumulation in the training loop for an effective larger batch.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return batch[0]
