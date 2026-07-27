#!/bin/bash

set -euo pipefail

# Five-epoch timing run for planned updates batch 03.
#
# This batch adds the parameter-matched CRPS FNO architecture ablation across
# all four main-comparison datasets.
# The resulting timing.ckpt is consumed by
# submit_planned_updates_03_large.sh to derive the number of epochs that fit
# the same 24h budget as the ViT baseline.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_updates_03"

declare -A EXPERIMENTS=(
    ["gray_scott"]="ablations/arch_unet_fno_vit/gray_scott/crps_fno_80m"
    ["gpe_laser_only_wake"]="ablations/arch_unet_fno_vit/gpe_laser_wake_only/crps_fno_80m"
    ["conditioned_navier_stokes"]="ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_fno_80m"
    ["advection_diffusion"]="ablations/arch_unet_fno_vit/advection_diffusion/crps_fno_80m"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="fno_m8_crps_${datamodule}"

    echo "Submitting planned updates batch 03 timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo "  run_id: ${run_id}"

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned updates batch 03 timing jobs submitted."
echo ""
echo "Once the SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/fno_m8_crps_*/retrieve.sh; do bash \"\$f\"; done"
