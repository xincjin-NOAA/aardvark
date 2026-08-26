"""
Vendored subset of ocelot3's raw-data loading/QC/normalization code, copied in
so this adapter runs standalone -- no sibling ocelot3 checkout or matching
Python version required.

Vendored from ocelot3 (branch feature/urma_obs_anemoi_core_nodes_aardvark, as of
2026-08-25/26): constants.py, logger.py, timing_utils.py, utils.py (DEFAULT_DTYPE
only), obs_config_utils.py, obs_config_urma.py, observation_config.py,
training/data/dataset_timeseries.py. Only import paths were changed
(`ocelot.x` -> `ocelot_vendor.x`) and `from __future__ import annotations` was
added to dataset_timeseries.py to fix a `seed: int | None` annotation that's
invalid syntax on Python < 3.10 (Aardvark's own conda env pins Python 3.8) --
no other logic was changed. If ocelot3's real pipeline changes (QC ranges,
normalization stats, instrument config), re-sync these files by hand; there is
no automated link back to the source repo.
"""
