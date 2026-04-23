#!/bin/bash

set -euo pipefail
# Timing run for MAE-initialized CRPS fine-tuning on CNS.
#
# Usage:
#   MAE_CHECKPOINT=/path/to/mae/encoder_processor_decoder.ckpt \
#     bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_timing.sh
#
# This fine-tune uses CRPS with n_members=16 and batch_size=16/GPU, keeping
# the effective global batch at 16 x 16 x 4 GPUs = 1024.

MAE_CHECKPOINT="${MAE_CHECKPOINT:-${1:-}}"
if [[ -z "${MAE_CHECKPOINT}" ]]; then
    echo "FATAL: set MAE_CHECKPOINT or pass the MAE checkpoint path as argv[1]" >&2
    exit 1
fi

if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
    echo "FATAL: MAE_CHECKPOINT does not exist: ${MAE_CHECKPOINT}" >&2
    exit 1
fi

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
)

CRPS_BUDGET_HOURS="${CRPS_BUDGET_HOURS:-4}"
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_vit_mae_to_crps"
N_MEMBERS=16
BS_PER_GPU=16

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="vit_mae_to_crps_${datamodule}_m${N_MEMBERS}"

    echo "Submitting MAE-initialized CRPS timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  mae checkpoint: ${MAE_CHECKPOINT}"
    echo "  n_members: ${N_MEMBERS}"
    echo "  bs_per_gpu: ${BS_PER_GPU}"
    echo "  run_id: ${run_id}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${CRPS_BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${CRPS_BUDGET_HOURS}" \
        datamodule="${datamodule}" \
        local_experiment="${experiment}" \
        model.n_members="${N_MEMBERS}" \
        datamodule.batch_size="${BS_PER_GPU}" \
        +resume_from_checkpoint="${MAE_CHECKPOINT}" \
        +resume_weights_only=true

    echo ""
    echo "---"
    echo ""
done

echo "All MAE-initialized CRPS timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/vit_mae_to_crps_*/retrieve.sh; do bash \"\$f\"; done"
