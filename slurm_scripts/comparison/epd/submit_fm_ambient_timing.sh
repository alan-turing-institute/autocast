#!/bin/bash

set -euo pipefail
# Submit FM-in-ambient timing jobs for 4 target datasets.
# Model: flow_matching_vit (vit backbone, hid_channels=704, hid_blocks=12,
# attention_heads=8, patch_size=4, flow_ode_steps=50). Encoder/decoder:
# permute_concat + channels_last (architecture parity with ambient CRPS).
# Optimizer: adamw_half (LR=1e-4, warmup=0). Batch size: 256/GPU
# (effective-batch parity with CRPS bs=32 x n_members=8).
# See local_hydra/local_experiment/epd/<dataset>/fm_vit_large.yaml for the
# authoritative hyperparameters.
# Runs 5 epochs each to measure per-epoch wall-clock time, then use
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# to compute the max_epochs for a 24h budget.

# Per-dataset local_experiment configs.
declare -A EXPERIMENTS=(
    ["gray_scott"]="epd/gray_scott/fm_vit_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/fm_vit_large"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/fm_vit_large"
    ["advection_diffusion"]="epd/advection_diffusion/fm_vit_large"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"

    echo "Submitting FM-in-ambient timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "fm_ambient_b256_${datamodule}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}"

    echo ""
    echo "---"
    echo ""
done

echo "All timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/fm_ambient_b256_*/retrieve.sh; do bash \"\$f\"; done"
