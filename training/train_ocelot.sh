#!/bin/bash -l
#
# SLURM launcher for train_ocelot.py (Aardvark's architecture trained on real
# ocelot3 URMA DA data -- see ../OCELOT_ADAPTER.md). Styled after ocelot3's own
# job scripts (submit_experiments_from_config.sh) since this runs on the same
# cluster, but this is a SEPARATE conda env from ocelot3's -- see the env
# activation TODO below before submitting.
#
# Usage (submit from the aardvark repo root, not from training/):
#   mkdir -p jobs   # once, if it doesn't exist yet -- sbatch needs -o/-e dirs to pre-exist
#   sbatch training/train_ocelot.sh
#
# Override any config var at submit time, e.g. a quick 3-day smoke run:
#   sbatch --export=ALL,START_DATE=2025-02-01,END_DATE=2025-02-03,EPOCHS=1 training/train_ocelot.sh
#
#SBATCH -A gpu-emc-ai
#SBATCH -p u1-h100
#SBATCH -q gpu
#SBATCH --gres=gpu:h100:1          # train_ocelot.py is single-process (batch size fixed
#SBATCH -J aardvark_ocelot         # at 1 bin/step, see OCELOT_ADAPTER.md) -- 1 GPU, no DDP.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -o jobs/train_ocelot_%J.out
#SBATCH -e jobs/train_ocelot_%J.err
# -A/-p/-q/gres above are copied from ocelot3's submit_experiments_from_config.sh
# (same cluster) -- verify this account/partition/QOS is actually valid for
# aardvark work before relying on it; adjust -t (time limit) once you know how
# long a real epoch takes.

set -x
set -e

# === Environment ===
# IMPORTANT: this must activate AARDVARK's conda env (environment.yml calls it
# "npw", Python 3.8+torch+timm), NOT ocelot3's env_diatom.sh -- that activates
# ocelot3's own (Python 3.10+) env, and mixing the two up is exactly the
# Python-version mismatch that ocelot_vendor/ was built to route around (see
# ocelot_vendor/__init__.py). Fill in however conda is activated on this
# cluster for the aardvark/npw env, e.g.:
#   module load conda
#   conda activate npw
# Left unset on purpose -- I don't know this cluster's conda init path.
: "${AARDVARK_CONDA_ENV:?Set AARDVARK_CONDA_ENV (e.g. npw) or edit this script directly with your conda activation lines.}"
module load conda 2>/dev/null || true
conda activate "$AARDVARK_CONDA_ENV"

cd "$(dirname "$0")/../aardvark"

echo "=========================================="
echo "Running on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Conda env: $AARDVARK_CONDA_ENV"
echo "=========================================="
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# === Config -- override any of these via --export=ALL,VAR=value at submit time ===
DATA_PATH="${DATA_PATH:-/scratch5/purged/Xin.C.Jin/data/v1/diag_urma/}"
STATIC_DATA_DIR="${STATIC_DATA_DIR:-/scratch3/NCEPDEV/da/Xin.C.Jin/my_data/urma}"
START_DATE="${START_DATE:-2025-02-01}"
END_DATE="${END_DATE:-2025-02-03}"       # small window by default -- widen once a short run works
VAL_START_DATE="${VAL_START_DATE:-}"
VAL_END_DATE="${VAL_END_DATE:-}"
EPOCHS="${EPOCHS:-5}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
TARGET_SAMPLE_SIZE="${TARGET_SAMPLE_SIZE:-20000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./ocelot_checkpoints}"

VAL_ARGS=()
if [ -n "$VAL_START_DATE" ] && [ -n "$VAL_END_DATE" ]; then
    VAL_ARGS=(--val_start_date "$VAL_START_DATE" --val_end_date "$VAL_END_DATE")
fi

CMD=(python train_ocelot.py
    --data_path "$DATA_PATH"
    --static_data_dir "$STATIC_DATA_DIR"
    --start_date "$START_DATE"
    --end_date "$END_DATE"
    --epochs "$EPOCHS"
    --grad_accum_steps "$GRAD_ACCUM_STEPS"
    --target_sample_size "$TARGET_SAMPLE_SIZE"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --device cuda
    "${VAL_ARGS[@]}"
)

echo "=========================================="
echo "${CMD[@]}"
echo "=========================================="
"${CMD[@]}"
