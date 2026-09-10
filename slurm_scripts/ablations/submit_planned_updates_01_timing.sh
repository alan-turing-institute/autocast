#!/bin/bash

set -euo pipefail

# Five-epoch timing jobs for planned updates batch 01.
#
# This batch adds the four-dataset CRPS MC-dropout ablation, replacing
# stochastic modulation with FFN dropout at p=0.1. Run this before the
# production submitter so each dataset receives a timing-derived 24h schedule.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_updates_01"

declare -A EXPERIMENTS=(
    ["gray_scott"]="ablations/mc_dropout/gray_scott/crps_vit_azula_mc_dropout_large"
    ["gpe_laser_only_wake"]="ablations/mc_dropout/gpe_laser_wake_only/crps_vit_azula_mc_dropout_large"
    ["conditioned_navier_stokes"]="ablations/mc_dropout/conditioned_navier_stokes/crps_vit_azula_mc_dropout_large"
    ["advection_diffusion"]="ablations/mc_dropout/advection_diffusion/crps_vit_azula_mc_dropout_large"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="mc_dropout_crps_${datamodule}"

    echo "Submitting planned updates batch 01 timing run"
    echo "  run_id: ${run_id}"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        datamodule="${datamodule}" \
        local_experiment="${experiment}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned updates batch 01 timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/mc_dropout_crps_*/retrieve.sh; do bash \"\$f\"; done"
