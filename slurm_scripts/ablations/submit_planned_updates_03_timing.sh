#!/bin/bash

set -euo pipefail

# Five-epoch timing run for planned updates batch 03.
#
# This batch adds the parameter-matched CNS CRPS FNO architecture ablation.
# The resulting timing.ckpt is consumed by
# submit_planned_updates_03_large.sh to derive the number of epochs that fit
# the same 24h budget as the ViT baseline.

EXPERIMENT="ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_fno_80m"
BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_updates_03"
RUN_ID="fno_m8_crps_cns"

echo "Submitting planned updates batch 03 timing run"
echo "  local_experiment: ${EXPERIMENT}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"

uv run autocast time-epochs --kind epd --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}"

echo ""
echo "Once the SLURM job completes, retrieve its result with:"
echo "  bash outputs/${RUN_GROUP}/${RUN_ID}/retrieve.sh"
