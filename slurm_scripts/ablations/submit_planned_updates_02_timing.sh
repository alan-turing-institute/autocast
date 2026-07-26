#!/bin/bash

set -euo pipefail

# Five-epoch timing jobs for planned updates batch 02.
#
# This batch trains the CNS MC-dropout MSE + L2 baseline with p=0.1.
# Other reusable dataset configs remain listed below for easy re-enabling.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_updates_02"

declare -A EXPERIMENTS=(
    # ["gray_scott"]="ablations/mc_dropout/gray_scott/mse_vit_azula_mc_dropout_large"
    # ["gpe_laser_only_wake"]="ablations/mc_dropout/gpe_laser_wake_only/mse_vit_azula_mc_dropout_large"
    ["conditioned_navier_stokes"]="ablations/mc_dropout/conditioned_navier_stokes/mse_vit_azula_mc_dropout_large"
    # ["advection_diffusion"]="ablations/mc_dropout/advection_diffusion/mse_vit_azula_mc_dropout_large"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="mc_dropout_mse_${datamodule}"

    echo "Submitting planned updates batch 02 timing run"
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

echo "Planned updates batch 02 CNS timing job submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/mc_dropout_mse_*/retrieve.sh; do bash \"\$f\"; done"
