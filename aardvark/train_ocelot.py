"""
Standalone training entrypoint: Aardvark's ConvCNP+ViT architecture (via
ocelot_models.py) trained on ocelot3's real URMA DA data (via ocelot_loader.py,
using the vendored data pipeline in ocelot_vendor/ -- no ocelot3 checkout or
matching Python version needed), independent of both Aardvark's own
(non-functional) loader.py and ocelot3's PyG/mesh training stack.

Example:
    python train_ocelot.py \
        --data_path /scratch5/purged/Xin.C.Jin/data/v1/diag_urma/ \
        --start_date 2025-02-01 --end_date 2025-02-20 \
        --val_start_date 2025-02-21 --val_end_date 2025-02-28 \
        --device cuda

Notes:
  - Batch size is fixed at 1 bin/step (see ocelot_loader.collate_single for why);
    use --grad_accum_steps for an effective larger batch.
  - The default --grid_nlon/--grid_nlat/--patch_size/--vit_depth are much smaller
    than Aardvark's own global-model defaults (240x121, depth 16) since this is a
    CONUS-only domain and this adapter hasn't been tuned for scale yet -- treat
    them as a starting point, not a validated configuration.
"""
import argparse
import os
import time

import torch

from ocelot_loader import OcelotWeatherDataset, collate_single
from ocelot_models import OcelotAardvarkModel, masked_mse_loss

# Real CONUS URMA run bounds (lon in 0-360 convention), from jobs/bak/train.out
# in ocelot3 -- override with --lon_min/--lon_max/--lat_min/--lat_max for a
# different domain.
DEFAULT_LON_MIN = 233.723448
DEFAULT_LON_MAX = 300.957852061
DEFAULT_LAT_MIN = 18.228976
DEFAULT_LAT_MAX = 56.37275379
DEFAULT_STATIC_DATA_DIR = "/scratch3/NCEPDEV/da/Xin.C.Jin/my_data/urma"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_path", required=True, help="ocelot3 URMA Parquet data_path (per-instrument obs).")
    p.add_argument("--static_data_dir", default=DEFAULT_STATIC_DATA_DIR)
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date", required=True)
    p.add_argument("--val_start_date", default=None)
    p.add_argument("--val_end_date", default=None)
    p.add_argument("--delta_time", type=int, default=1)
    p.add_argument("--obs_config", default="urma")
    p.add_argument("--exp_type", default="regional_da")
    p.add_argument("--target_is_anal", action="store_true")
    p.add_argument("--normalize_increment", action="store_true")
    p.add_argument("--target_sample_size", type=int, default=20000)

    p.add_argument("--lon_min", type=float, default=DEFAULT_LON_MIN)
    p.add_argument("--lon_max", type=float, default=DEFAULT_LON_MAX)
    p.add_argument("--lat_min", type=float, default=DEFAULT_LAT_MIN)
    p.add_argument("--lat_max", type=float, default=DEFAULT_LAT_MAX)
    p.add_argument("--grid_nlon", type=int, default=96)
    p.add_argument("--grid_nlat", type=int, default=64)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--vit_embed_dim", type=int, default=256)
    p.add_argument("--vit_depth", type=int, default=6)
    p.add_argument("--vit_num_heads", type=int, default=8)
    p.add_argument("--int_channels", type=int, default=64)
    p.add_argument("--mlp_h_channels", type=int, default=128)
    p.add_argument("--mlp_h_layers", type=int, default=3)
    p.add_argument("--init_ls", type=float, default=0.001)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--checkpoint_dir", default="./ocelot_checkpoints")
    p.add_argument("--save_val_outputs", action="store_true",
                    help="Dump y_hat/y_target/valid_mask per val bin to <checkpoint_dir>/val_outputs/epoch_<N>/.")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def move_task_to_device(task, device):
    out = {}
    for k, v in task.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        elif isinstance(v, list) and len(v) == 2 and all(torch.is_tensor(t) for t in v):
            out[k] = [v[0].to(device), v[1].to(device)]
        else:
            out[k] = v
    return out


def reconstruct_analysis(y_hat, task, dataset: OcelotWeatherDataset):
    """anal_norm = ges + increment, undoing --normalize_increment rescaling if used."""
    if dataset.target_is_anal:
        return y_hat
    if dataset.normalize_increment:
        y_hat = y_hat * dataset.increment_std.to(y_hat.device) + dataset.increment_mean.to(y_hat.device)
    return task["ges_at_target"].to(y_hat.device) + y_hat


def build_dataset(args, start_date, end_date):
    return OcelotWeatherDataset(
        data_path=args.data_path,
        static_data_dir=args.static_data_dir,
        start_date=start_date,
        end_date=end_date,
        delta_time=args.delta_time,
        obs_config_name=args.obs_config,
        exp_type=args.exp_type,
        target_is_anal=args.target_is_anal,
        normalize_increment=args.normalize_increment,
        target_sample_size=args.target_sample_size,
        seed=args.seed,
    )


@torch.no_grad()
def evaluate(model, val_loader, device, save_dir=None):
    """Runs validation, printing per-bin loss. If save_dir is set, also dumps
    y_hat/y_target/valid_mask (normalized, same space as the loss) per bin to
    "<save_dir>/<bin_name>.pt" for later inspection."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for task in val_loader:
        if task is None:
            continue
        task = move_task_to_device(task, device)
        y_hat = model(task)
        loss = masked_mse_loss(y_hat, task["y_target"], task["y_target_valid_mask"])
        total_loss += loss.item()
        n_batches += 1
        bin_name = task.get("bin_name")
        print(f"  val bin {bin_name} loss {loss.item():.6f}")
        if save_dir is not None:
            torch.save(
                {
                    "bin_name": bin_name,
                    "loss": loss.item(),
                    "y_hat": y_hat.detach().cpu(),
                    "y_target": task["y_target"].detach().cpu(),
                    "y_target_valid_mask": task["y_target_valid_mask"].detach().cpu(),
                },
                os.path.join(save_dir, f"{bin_name}.pt"),
            )
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = build_dataset(args, args.start_date, args.end_date)
    val_dataset = None
    if args.val_start_date and args.val_end_date:
        val_dataset = build_dataset(args, args.val_start_date, args.val_end_date)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_single,
        num_workers=args.num_workers,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, collate_fn=collate_single, num_workers=args.num_workers
        )

    model = OcelotAardvarkModel(
        instrument_channels=train_dataset.instrument_channel_dims(),
        target_dim=train_dataset.target_dim,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        grid_nlon=args.grid_nlon,
        grid_nlat=args.grid_nlat,
        patch_size=args.patch_size,
        vit_embed_dim=args.vit_embed_dim,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        int_channels=args.int_channels,
        mlp_h_channels=args.mlp_h_channels,
        mlp_h_layers=args.mlp_h_layers,
        init_ls=args.init_ls,
        device=device,
    ).to(device)

    print(f"Instrument channels: {train_dataset.instrument_channel_dims()}")
    print(f"Encoder output channels: {model.encoder.out_channels}")
    print(f"Train bins: {len(train_dataset)}" + (f", val bins: {len(val_dataset)}" if val_dataset else ""))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        optimizer.zero_grad()
        running_loss = 0.0
        for i, task in enumerate(train_loader):
            if task is None:
                continue
            task = move_task_to_device(task, device)
            y_hat = model(task)
            loss = masked_mse_loss(y_hat, task["y_target"], task["y_target_valid_mask"])
            (loss / args.grad_accum_steps).backward()
            running_loss += loss.item()
            step += 1

            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} bin {task.get('bin_name')} loss {loss.item():.6f}")

        avg_loss = running_loss / max(i + 1, 1)
        msg = f"epoch {epoch} done in {time.time() - epoch_start:.1f}s, avg train loss {avg_loss:.6f}"
        if val_loader is not None:
            val_save_dir = (
                os.path.join(args.checkpoint_dir, "val_outputs", f"epoch_{epoch}")
                if args.save_val_outputs else None
            )
            val_loss = evaluate(model, val_loader, device, save_dir=val_save_dir)
            msg += f", val loss {val_loss:.6f}"
        print(msg)

        ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch}.pt")
        torch.save(
            {"model_state_dict": model.state_dict(), "args": vars(args)},
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
