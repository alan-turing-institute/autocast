#!/bin/bash

set -euo pipefail

# Timing jobs for the planned batch 02 m=4 CRPS follow-up.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_02_m4_followup"

# run_id|datamodule|local_experiment
RUNS=(
    "vit_m4_gpe|gpe_laser_only_wake|epd/gpe_laser_wake_only/crps_vit_azula_large"
    "vit_m4_ad|advection_diffusion|epd/advection_diffusion/crps_vit_azula_large"
)

run_overrides() {
    local datamodule="$1"

    printf '%s\n' \
        "datamodule=${datamodule}" \
        "model.n_members=4" \
        "datamodule.batch_size=64"
}

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id datamodule experiment <<< "${run_spec}"
    mapfile -t overrides < <(run_overrides "${datamodule}")

    echo "Submitting planned batch 02 m=4 follow-up timing run"
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
        local_experiment="${experiment}" \
        "${overrides[@]}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned batch 02 m=4 follow-up timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/*/retrieve.sh; do bash \"\$f\"; done"
