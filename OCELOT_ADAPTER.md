# ocelot3 data adapter

Lets Aardvark's ConvCNP+ViT architecture run standalone, trained on real URMA
DA data read directly from ocelot3's Parquet pipeline, instead of Aardvark's
own `loader.py` (which points at placeholder paths and was never runnable).

New files, all under `aardvark/`:
- `ocelot_loader.py` -- `OcelotWeatherDataset`, reusing ocelot3's
  `ParquetDataManager`/`extract_features_from_df`/`obs_config_urma.py` as-is.
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

## Known limitations (read before trusting results)

- **Not executed.** Written and reasoned through by hand against both
  codebases -- no Python/torch was available on the machine this was written
  on. Run `test_ocelot_pipeline.py` in your real conda env (the one with
  torch/timm/pandas/pyarrow/scikit-learn plus `ocelot3` importable) before
  trusting anything else here.
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
python test_ocelot_pipeline.py --ocelot_repo_path /path/to/ocelot3
```

## Running real training

```
python train_ocelot.py \
    --ocelot_repo_path /path/to/ocelot3 \
    --data_path /scratch5/purged/Xin.C.Jin/data/v1/diag_urma/ \
    --start_date 2025-02-01 --end_date 2025-02-20 \
    --val_start_date 2025-02-21 --val_end_date 2025-02-28 \
    --device cuda
```
