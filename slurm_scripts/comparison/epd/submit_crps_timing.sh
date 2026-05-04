#!/bin/bash

set -euo pipefail
# Submit CRPS-in-ambient timing jobs for 4 target datasets.
# Model: vit_azula_large (hidden_dim=568, n_layers=12, num_heads=8,
# patch_size=4, n_noise_channels=1024). Optimizer: adamw_half (LR=2e-4,
# warmup=0). Batch size: 32/GPU (x4 GPUs x n_members=8 = 1024 effective).
# See local_hydra/local_experiment/epd/<dataset>/crps_vit_azula_large.yaml
# for the authoritative hyperparameters.
# Runs 5 epochs each to measure per-epoch wall-clock time, then use
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# to compute the max_epochs for a 24h budget.

# Per-dataset local_experiment configs.
declare -A EXPERIMENTS=(
    ["gray_scott"]="epd/gray_scott/crps_vit_azula_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
    ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"

    echo "Submitting CRPS timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "crps_b32_${datamodule}" \
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
echo "  for f in outputs/${RUN_GROUP}/crps_b32_*/retrieve.sh; do bash \"\$f\"; done"
