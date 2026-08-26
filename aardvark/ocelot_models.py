"""
Generalized ConvCNP encoder/processor/decoder for running Aardvark's architecture
against ocelot3's URMA DA data (see ocelot_loader.py for the data side).

Reuses Aardvark's `convDeepSet` (set_convs.py) and `ViT` (vit.py) unmodified --
those are the two architecturally load-bearing pieces (the set-convolution
regridding math and the ViT processor). What's new here, relative to
`models.ConvCNPWeather` / `misc_downscaling_functionality.ConvCNPWeatherOnToOff`,
is that the encoder's per-modality methods (encoder_hadisd, encoder_amsua, ...)
are replaced by one config-driven loop over ocelot3's instrument list, and the
OnToOff decoder targets ocelot3's arbitrary analysis-grid query points instead
of a fixed HadISD station set.

Note: `models.ConvCNPWeather.__init__` hardcodes `.cuda()` when building its
internal grid, which would crash on a CPU-only machine. The grid tensors here
are registered as buffers instead, so `.to(device)` on the whole model moves
them like any other module -- this only matters for local CPU testing
(see test_ocelot_pipeline.py); real training still wants a GPU.
"""
from typing import Dict

import torch
import torch.nn as nn

from set_convs import convDeepSet
from vit import ViT


class OcelotConvCNPEncoder(nn.Module):
    """
    Config-driven analog of `models.ConvCNPWeather`'s encoder half: one
    `convDeepSet` per channel per instrument (matching Aardvark's per-variable
    learned-lengthscale convention), all regridding onto a shared internal grid.

    Every ocelot3 instrument (including "state_ges", the URMA background +
    terrain/slmask) is genuinely irregular point data, so everything goes
    through convDeepSet's "OffToOn" mode -- unlike Aardvark's own AMSU/HIRS/
    GRIDSAT encoders, which use "OnToOn" because that data arrives pre-gridded.
    """

    def __init__(
        self,
        instrument_channels: Dict[str, int],
        grid_lon: torch.Tensor,
        grid_lat: torch.Tensor,
        init_ls: float = 0.001,
        device=None,
    ):
        super().__init__()
        self.instrument_channels = dict(instrument_channels)
        self.register_buffer("grid_lon", grid_lon.unsqueeze(0).float())
        self.register_buffer("grid_lat", grid_lat.unsqueeze(0).float())

        self.setconvs = nn.ModuleDict()
        for inst_key, n_channels in self.instrument_channels.items():
            self.setconvs[inst_key] = nn.ModuleList(
                [
                    convDeepSet(init_ls, "OffToOn", device, density_channel=True)
                    for _ in range(n_channels)
                ]
            )

    @property
    def out_channels(self) -> int:
        # +1 density channel per real channel, matching convDeepSet(density_channel=True).
        return sum(n * 2 for n in self.instrument_channels.values())

    def forward(self, task: Dict) -> torch.Tensor:
        int_grid = [self.grid_lon, self.grid_lat]
        b = int_grid[0].shape[0]
        device = int_grid[0].device

        encodings = []
        for inst_key, n_channels in self.instrument_channels.items():
            x_in = task.get(f"{inst_key}_x_current")
            wt = task.get(f"{inst_key}_current")
            convs = self.setconvs[inst_key]
            if x_in is None or wt is None:
                # No obs for this instrument in this bin: feed convDeepSet a
                # zero-length obs set rather than special-casing the output --
                # it naturally produces an all-zero, zero-density encoding.
                x_lon = torch.zeros(b, 0, device=device)
                x_lat = torch.zeros(b, 0, device=device)
                wt = torch.zeros(b, n_channels, 0, device=device)
            else:
                x_lon, x_lat = x_in
                x_lon = x_lon.to(device)
                x_lat = x_lat.to(device)
                wt = wt.to(device)
            for c in range(n_channels):
                encodings.append(
                    convs[c](
                        x_in=[x_lon.clone(), x_lat.clone()],
                        wt=wt[:, c : c + 1, :],
                        x_out=int_grid,
                    )
                )
        return torch.cat(encodings, dim=1)


class ResidualBlock(nn.Module):
    """Same design as `misc_downscaling_functionality.ResidualBlock` (re-declared locally
    to avoid importing that module's Unet-decoder dependencies, which this model doesn't use)."""

    def __init__(self, n_channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(n_channels, n_channels), nn.ReLU())

    def forward(self, x):
        return self.block(x) + x


class PointDecoderMLP(nn.Module):
    """Same design as `misc_downscaling_functionality.DownscalingMLP`."""

    def __init__(self, in_channels: int, out_channels: int, h_channels: int = 128, h_layers: int = 3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, h_channels),
            *[ResidualBlock(h_channels) for _ in range(h_layers)],
            nn.Linear(h_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class OcelotOnToOffDecoder(nn.Module):
    """
    Config-driven analog of `ConvCNPWeatherOnToOff`: dense processed grid ->
    arbitrary query points via convDeepSet "OnToOff", + a small MLP head that
    also sees static terrain/slmask and lon/lat at each query point.

    Aardvark's own decoder targets a fixed few-thousand HadISD stations; here
    the query points are ocelot3's analysis-grid cells (up to ~3.7M/bin), so
    callers should chunk queries for full-grid inference -- see
    OcelotAardvarkModel.predict_full_grid.
    """

    def __init__(
        self,
        int_channels: int,
        target_dim: int,
        grid_lon: torch.Tensor,
        grid_lat: torch.Tensor,
        mlp_h_channels: int = 128,
        mlp_h_layers: int = 3,
        init_ls: float = 0.001,
        device=None,
    ):
        super().__init__()
        self.register_buffer("grid_lon", grid_lon.unsqueeze(0).float())
        self.register_buffer("grid_lat", grid_lat.unsqueeze(0).float())
        self.sc_out = convDeepSet(init_ls, "OnToOff", device, density_channel=False)
        self.mlp = PointDecoderMLP(
            in_channels=int_channels + 2 + 2,  # decoded channels + [terrain, slmask] + [lon, lat]
            out_channels=target_dim,
            h_channels=mlp_h_channels,
            h_layers=mlp_h_layers,
        )

    def forward(self, processed_grid: torch.Tensor, x_target: torch.Tensor, static_at_target: torch.Tensor):
        """
        processed_grid: (B, n_lon, n_lat, int_channels), ViT output (channel-last).
        x_target: (B, 2, M) lon/lat of query points, scaled by /360.
        static_at_target: (B, 2, M) terrain/slmask at query points.
        -> (B, M, target_dim)
        """
        x = processed_grid.permute(0, 3, 1, 2).contiguous()  # (B, C, n_lon, n_lat)
        decoded = self.sc_out(
            x_in=[self.grid_lon.clone(), self.grid_lat.clone()],
            wt=x,
            x_out=[x_target[:, 0, :].clone(), x_target[:, 1, :].clone()],
        )  # (B, C, M)
        feats = torch.cat([decoded, static_at_target, x_target], dim=1)  # (B, C+4, M)
        return self.mlp(feats.permute(0, 2, 1))  # (B, M, target_dim)


class OcelotAardvarkModel(nn.Module):
    """Encoder -> ViT processor -> OnToOff decoder, end to end."""

    def __init__(
        self,
        instrument_channels: Dict[str, int],
        target_dim: int,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        grid_nlon: int = 96,
        grid_nlat: int = 64,
        patch_size: int = 8,
        vit_embed_dim: int = 256,
        vit_depth: int = 6,
        vit_num_heads: int = 8,
        int_channels: int = 64,
        mlp_h_channels: int = 128,
        mlp_h_layers: int = 3,
        init_ls: float = 0.001,
        device=None,
    ):
        super().__init__()
        if grid_nlon % patch_size != 0 or grid_nlat % patch_size != 0:
            raise ValueError(
                f"grid_nlon ({grid_nlon}) and grid_nlat ({grid_nlat}) must both be evenly "
                f"divisible by patch_size ({patch_size}) -- required by timm's PatchEmbed."
            )

        grid_lon = torch.linspace(lon_min, lon_max, grid_nlon) / 360.0
        grid_lat = torch.linspace(lat_min, lat_max, grid_nlat) / 360.0

        self.encoder = OcelotConvCNPEncoder(instrument_channels, grid_lon, grid_lat, init_ls=init_ls, device=device)
        self.processor = ViT(
            in_channels=self.encoder.out_channels,
            out_channels=int_channels,
            h_channels=vit_embed_dim,
            img_size=[grid_nlon, grid_nlat],
            patch_size=patch_size,
            depth=vit_depth,
            num_heads=vit_num_heads,
            per_var_embedding=True,
        )
        self.decoder = OcelotOnToOffDecoder(
            int_channels=int_channels,
            target_dim=target_dim,
            grid_lon=grid_lon,
            grid_lat=grid_lat,
            mlp_h_channels=mlp_h_channels,
            mlp_h_layers=mlp_h_layers,
            init_ls=init_ls,
            device=device,
        )

    def encode_and_process(self, task: Dict) -> torch.Tensor:
        x = self.encoder(task)
        return self.processor(x, lead_times=task["lt"])

    def forward(self, task: Dict) -> torch.Tensor:
        processed = self.encode_and_process(task)
        return self.decoder(processed, task["x_target"], task["static_at_target"])

    @torch.no_grad()
    def predict_full_grid(
        self,
        task: Dict,
        lon_all: torch.Tensor,
        lat_all: torch.Tensor,
        static_all: torch.Tensor,
        chunk_size: int = 20000,
    ) -> torch.Tensor:
        """
        Decode every point in (lon_all, lat_all) in chunks, for full-grid eval/inference
        rather than the random per-batch subsample used in training.
        lon_all/lat_all: (N,). static_all: (N, 2).
        """
        processed = self.encode_and_process(task)
        device = processed.device
        n = lon_all.shape[0]
        preds = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x_target = torch.stack([lon_all[start:end], lat_all[start:end]], dim=0).unsqueeze(0).to(device)
            static_chunk = static_all[start:end].t().unsqueeze(0).to(device)
            preds.append(self.decoder(processed, x_target, static_chunk).squeeze(0))
        return torch.cat(preds, dim=0)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid entries only (mirrors ocelot3's valid_mask-gated loss)."""
    valid_mask = valid_mask.bool()
    if valid_mask.sum() == 0:
        return pred.sum() * 0.0
    diff = (pred - target) ** 2
    return diff[valid_mask].mean()
