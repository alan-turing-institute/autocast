#!/bin/bash

set -euo pipefail
# CNS-only AE compression ablation — timing run.
# Trains the 4-level (64,128,256,512) encoder/decoder on CNS for 5 epochs
# to fit a 24h cosine schedule via `autocast time-epochs`.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/ae_compression_timing"

datamodule="conditioned_navier_stokes"
experiment="ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8"

echo "Submitting AE compression timing run"
echo "  datamodule: ${datamodule}"
echo "  local_experiment: ${experiment}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"

uv run autocast time-epochs --kind ae --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "ae_b16_${datamodule}_f8" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${experiment}"

echo ""
echo "Once SLURM job completes, retrieve with:"
echo "  bash outputs/${RUN_GROUP}/ae_b16_${datamodule}_f8/retrieve.sh"
echo "Then extract cosine_epochs:"
echo "  uv run autocast time-epochs --from-checkpoint outputs/${RUN_GROUP}/ae_b16_${datamodule}_f8/timing.ckpt -b ${BUDGET_HOURS}"
