# Understanding the Aardvark Weather Model

This is a suggested reading path through the codebase for the Aardvark Weather
model (https://www.nature.com/articles/s41586-025-08897-0) — a three-stage
encoder/processor/decoder weather forecasting system built on ConvCNP and ViT
architectures.

## 1. Start with the paper + README

[README.md](README.md) explains the three training stages (encoder →
processor → decoder) and end-to-end finetuning. Read the linked Nature paper
for the underlying math/motivation, since the code itself has minimal
comments.

## 2. Run the demo notebooks first, before reading source

These are the fastest way to see data shapes and outputs concretely:

- [notebooks/data_demo.ipynb](notebooks/data_demo.ipynb) — shows what one
  timeslice of input data looks like
- [notebooks/forecast_demo.ipynb](notebooks/forecast_demo.ipynb) — loads the
  full model, runs inference, visualizes forecasts
- [notebooks/e2e_finetune_demo.ipynb](notebooks/e2e_finetune_demo.ipynb) —
  end-to-end finetuning demo

## 3. Read the core building blocks (small, self-contained)

- [aardvark/architectures.py](aardvark/architectures.py) — just an `MLP`
  class, trivial
- [aardvark/set_convs.py](aardvark/set_convs.py) — `convDeepSet`, the ConvCNP
  set-convolution used to map irregular observations onto a grid
- [aardvark/vit.py](aardvark/vit.py) — Vision Transformer used as the
  processor backbone
- [aardvark/unet_wrap_padding.py](aardvark/unet_wrap_padding.py) — U-Net
  components with longitude wrap-around padding (handles the globe's periodic
  boundary)

## 4. Read the three main modules in aardvark/models.py

- `ConvCNPWeather` ([models.py:15](aardvark/models.py)) is used for **both
  encoder and processor** — it combines the set-conv (irregular obs → grid)
  with a U-Net/ViT backbone. The `mode` param (`"assimilation"` vs forecast)
  and `res` distinguish the two roles.
- Trace how `in_channels`/`out_channels`/`int_channels` differ between
  encoder and processor instantiations (check `training/train_encoder.sh` and
  `training/train_processor.sh` for the actual args).

## 5. Read the decoder

[aardvark/misc_downscaling_functionality.py](aardvark/misc_downscaling_functionality.py)
— `ConvCNPWeatherOnToOff`, which downscales gridded processor output to
station-level (HadISD) predictions.

## 6. Read the assembly point

[aardvark/e2e_model.py](aardvark/e2e_model.py) — `ConvCNPWeatherE2E` chains
encoder → `lead_time`-many processor steps → decoder, plus normalization.
This is the clearest single place to see how the three trained modules
compose into a full forecast.

## 7. Then the training machinery

Roughly in order of how central it is:

- [aardvark/train_module.py](aardvark/train_module.py) — generic training
  loop shared by encoder/processor
- [aardvark/finetune.py](aardvark/finetune.py) — processor finetuning
- [aardvark/e2e_train.py](aardvark/e2e_train.py) — end-to-end finetuning
  (pairs with `e2e_model.py`)
- [aardvark/trainer.py](aardvark/trainer.py) (602 lines) — top-level trainer
  orchestrating the above; worth reading last since it'll make more sense
  once you know what it's driving
- [aardvark/loss_functions.py](aardvark/loss_functions.py) — what's actually
  being optimized at each stage

## 8. Data loading last

[aardvark/loader.py](aardvark/loader.py) is the largest file (1825 lines) and
mostly plumbing; skim it only once you know what shapes the model expects
(from step 2's data_demo notebook) and what `data_shapes.py` documents.
