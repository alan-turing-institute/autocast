#!/bin/bash

set -euo pipefail
# Timing run for deterministic ViT MAE pretraining on CNS.
#
# This starts from the CRPS ViT ambient architecture but disables the ensemble
# path (n_members=1) and trains with torch.nn.L1Loss. Run this first, then
# derive the 24h cosine schedule from timing.ckpt:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="ablations/vit_mae_pretrain/conditioned_navier_stokes/vit_azula_large_mae_no_ensemble"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_vit_mae_pretrain"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="vit_mae_pretrain_${datamodule}"

    echo "Submitting ViT MAE pretrain timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  run_id: ${run_id}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

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

echo "All ViT MAE pretrain timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/vit_mae_pretrain_*/retrieve.sh; do bash \"\$f\"; done"
