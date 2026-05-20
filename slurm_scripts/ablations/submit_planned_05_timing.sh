#!/bin/bash

set -euo pipefail

# Timing jobs for planned ablation batch 05.
#
# This batch mirrors planned_04, but uses plain CRPS loss for the three
# remaining raw-field EPD datasets: GS, GPE, and AD.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_05"

# run_id|kind|datamodule|local_experiment
RUNS=(
    "plain_crps_m8_gs|epd|gray_scott|ablations/crps_variants/gray_scott/crps_vit_plain"
    "plain_crps_m8_gpe|epd|gpe_laser_only_wake|ablations/crps_variants/gpe_laser_wake_only/crps_vit_plain"
    "plain_crps_m8_ad|epd|advection_diffusion|ablations/crps_variants/advection_diffusion/crps_vit_plain"
)

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id kind datamodule experiment <<< "${run_spec}"

    echo "Submitting planned batch 05 timing run"
    echo "  run_id: ${run_id}"
    echo "  kind: ${kind}"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"

    uv run autocast time-epochs --kind "${kind}" --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}" \
        datamodule="${datamodule}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned batch 05 timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/*/retrieve.sh; do bash \"\$f\"; done"
