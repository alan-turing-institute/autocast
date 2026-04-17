#!/bin/bash

set -euo pipefail
# Submit AE timing jobs for 4 target datasets.
# Uses per-dataset local_experiment configs which set the correct
# periodic/non-periodic boundary conditions.
# Runs 5 epochs each to measure per-epoch wall-clock time, then use
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# to compute the max_epochs for a 24h budget.

# Per-dataset local_experiment configs (set periodic BCs correctly).
declare -A EXPERIMENTS=(
    ["gray_scott"]="ae/gray_scott/ae_dc_large"
    ["gpe_laser_only_wake"]="ae/gpe_laser_wake_only/ae_dc_large"
    ["conditioned_navier_stokes"]="ae/conditioned_navier_stokes/ae_dc_large"
    ["advection_diffusion"]="ae/advection_diffusion/ae_dc_large"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"

    echo "Submitting AE timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind ae --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "ae_b16_${datamodule}" \
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
echo "  for f in outputs/${RUN_GROUP}/ae_b16_*/retrieve.sh; do bash \"\$f\"; done"
