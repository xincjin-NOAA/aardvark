#!/bin/bash
#
# Convenience wrapper around `sbatch training/train_ocelot.sh` that reads an
# experiment's config from a YAML file, names the run, and namespaces its
# SLURM log files by that name -- so multiple experiments submitted around
# the same time don't get jumbled together in jobs/ or under a shared
# checkpoint dir (train_ocelot.py nests checkpoints/val_outputs/loss_log.csv
# under $CHECKPOINT_DIR/<run_name>/ whenever --run_name is set).
#
# Usage (submit from the aardvark repo root):
#   ./submit_train_ocelot.sh <config.yaml> [VAR=value ...]
#
# Examples:
#   ./submit_train_ocelot.sh configs/train_ocelot_example.yaml
#   ./submit_train_ocelot.sh configs/train_ocelot_example.yaml EPOCHS=1
#
# See configs/train_ocelot_example.yaml for the config schema -- keys map
# onto train_ocelot.sh's config vars (case-insensitive) and must include
# run_name. Any VAR=value pairs given on the command line override the
# config file, and both are forwarded to train_ocelot.sh via --export, same
# as calling sbatch directly (see train_ocelot.sh's own header for the full
# list of overridable vars).

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <config.yaml> [VAR=value ...]" >&2
    exit 1
fi

CONFIG_FILE="$1"
shift

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$REPO_ROOT/jobs"   # sbatch needs -o/-e dirs to pre-exist

# Flatten the YAML config into VAR=value lines (uppercased keys, matching
# train_ocelot.sh's env vars). Requires pyyaml, already in the aardvark conda env.
mapfile -t CONFIG_VARS < <(python3 - "$CONFIG_FILE" <<'EOF'
import sys
import yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f) or {}
for k, v in cfg.items():
    if v is None:
        continue
    print(f"{k.upper()}={v}")
EOF
)

# Merge config vars with CLI overrides (CLI wins on key collisions).
declare -A VARS
for kv in "${CONFIG_VARS[@]}" "$@"; do
    VARS["${kv%%=*}"]="${kv#*=}"
done

RUN_NAME="${VARS[RUN_NAME]}"
if [ -z "$RUN_NAME" ]; then
    echo "Config file must set 'run_name' (or pass RUN_NAME=... on the command line)." >&2
    exit 1
fi

EXPORT_VARS="ALL"
for key in "${!VARS[@]}"; do
    EXPORT_VARS="$EXPORT_VARS,$key=${VARS[$key]}"
done

echo "Submitting run '$RUN_NAME' with: $EXPORT_VARS"
sbatch \
    --job-name "aardvark_${RUN_NAME}" \
    --output "$REPO_ROOT/jobs/train_aardvark_${RUN_NAME}_%J.out" \
    --error "$REPO_ROOT/jobs/train_aardvark_${RUN_NAME}_%J.err" \
    --export="$EXPORT_VARS" \
    "$REPO_ROOT/training/train_ocelot.sh"
