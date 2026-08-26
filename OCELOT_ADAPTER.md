# ocelot3 data adapter

Lets Aardvark's ConvCNP+ViT architecture run standalone, trained on real URMA
DA data, instead of Aardvark's own `loader.py` (which points at placeholder
paths and was never runnable). No ocelot3 checkout is needed at run time --
the pieces of ocelot3's data pipeline this depends on are vendored in.

New files, all under `aardvark/`:
- `ocelot_vendor/` -- vendored copy of ocelot3's `ParquetDataManager`/
  `extract_features_from_df`/`obs_config_urma.py` and their small dependencies
  (see `ocelot_vendor/__init__.py` for exactly what was copied and the two
  deliberate deviations from the original: a `from __future__ import
  annotations` fix for a Python-3.10-only `int | None` annotation that crashes
  on Aardvark's Python-3.8 conda env, and two confirmed-dead imports dropped).
- `ocelot_loader.py` -- `OcelotWeatherDataset`, built on the vendored pipeline.
- `ocelot_models.py` -- `OcelotAardvarkModel` (encoder -> ViT processor ->
  OnToOff decoder), reusing Aardvark's `convDeepSet` and `ViT` unmodified but
  driven by ocelot3's config-driven instrument list instead of Aardvark's
  hardcoded per-modality methods.
- `train_ocelot.py` -- standalone training CLI.
- `test_ocelot_pipeline.py` -- synthetic-data smoke test (schema/shape only,
  not a correctness/accuracy check).

## Design, in one paragraph

ocelot3 has no dense-grid loading step -- even the URMA background/analysis
field is loaded as flat point rows -- so every instrument (diag_t, diag_q,
diag_uv, diag_ps, and the background "state_ges", each from
`obs_config_urma.py`) is regridded onto a CONUS-bounded internal grid via
`convDeepSet`'s `OffToOn` mode, processed by Aardvark's `ViT`, then decoded
back down to arbitrary query points -- ocelot3's analysis-grid cells -- via
`convDeepSet`'s `OnToOff` mode, the same mechanism Aardvark uses to decode to
HadISD stations. The training target is ocelot3's DA increment
(`anal_norm - ges_norm`), matching what ocelot3's own model predicts.

## Keeping the vendored copy in sync

`ocelot_vendor/` is a manual snapshot, not a live link -- if ocelot3's real
QC ranges, normalization stats, `obs_config_urma.py` instrument list, or
`ParquetDataManager`/`extract_features_from_df` logic change, re-copy the
affected file(s) by hand and reapply the two deviations noted in
`ocelot_vendor/__init__.py`. There's no automation checking these have drifted.

## Known limitations (read before trusting results)

- **Not executed by the model that wrote it.** The first real run (on the
  actual HPC env) caught two bugs static reading missed: a wrong
  `--ocelot_repo_path` (now moot -- see above) and the Python-3.8/3.10 syntax
  mismatch fixed by vendoring. Rerun `test_ocelot_pipeline.py` after any
  further change before trusting it.
- **Batch size is fixed at 1 bin/step.** Each instrument has a different N per
  bin; `convDeepSet`'s `OffToOn` path takes N as a plain tensor dim with no
  cross-sample padding. Use `--grad_accum_steps` for an effective larger batch.
- **~3.7M analysis-grid cells/bin is too many query points to decode at once**
  during training; `target_sample_size` (default 20000) subsamples per step.
  For full-grid eval/inference use `OcelotAardvarkModel.predict_full_grid`,
  which chunks the `OnToOff` decode.
- **Grid resolution/model size defaults (96x64 internal grid, ViT depth 6) are
  arbitrary starting points**, picked to be CONUS-appropriately smaller than
  Aardvark's global 240x121/depth-16 defaults -- not validated against any
  training run.
- No lead-time/time-of-day conditioning is fed to the decoder (URMA DA is
  single-time, so this matters less than for forecasting, but per-obs time
  features that ocelot3 computes are currently unused by this adapter).

## Running the smoke test

```
python test_ocelot_pipeline.py
```

## Running real training

Interactively (e.g. on an interactive GPU allocation):
```
python train_ocelot.py \
    --data_path /scratch5/purged/Xin.C.Jin/data/v1/diag_urma/ \
    --start_date 2025-02-01 --end_date 2025-02-20 \
    --val_start_date 2025-02-21 --val_end_date 2025-02-28 \
    --device cuda
```

Or submit as a SLURM job with `training/train_ocelot.sh` (mirrors ocelot3's own
`submit_experiments_from_config.sh` job-script style, since this runs on the
same cluster). It needs `AARDVARK_CONDA_ENV` set to however you activate
Aardvark's conda env (`environment.yml` calls it `npw`) -- deliberately left
unset in the script rather than guessed, since it must NOT be ocelot3's own
`env_diatom.sh` (different Python version -- see `ocelot_vendor/__init__.py`
for why that distinction matters here specifically). Also verify the
`-A`/`-p`/`-q`/`--gres` SBATCH directives (copied from ocelot3's job scripts)
are actually valid for aardvark work under your allocation before relying on
them.

```
mkdir -p jobs   # once, if it doesn't exist -- sbatch needs -o/-e dirs to pre-exist
sbatch --export=ALL,AARDVARK_CONDA_ENV=npw training/train_ocelot.sh

# override any config var at submit time, e.g. a quick smoke run:
sbatch --export=ALL,AARDVARK_CONDA_ENV=npw,START_DATE=2025-02-01,END_DATE=2025-02-03,EPOCHS=1 \
    training/train_ocelot.sh
```
