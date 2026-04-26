#!/bin/bash

set -euo pipefail
# Timing run for MAE-initialized CRPS fine-tuning on CNS.
#
# Usage:
#   bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_timing.sh
#   # or explicitly:
#   MAE_CHECKPOINT=/path/to/mae/run_dir \
#     bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_timing.sh
#
# This fine-tune uses CRPS with n_members=8 and batch_size=32/GPU, matching
# the baseline effective global batch of 32 x 8 x 4 GPUs = 1024. The lower
# learning rate is intentional for MAE-initialized fine-tuning.

MAE_SOURCE="${MAE_CHECKPOINT:-${MAE_RUN_DIR:-${1:-}}}"

select_best_val_checkpoint() {
    awk '
        {
            base = $0
            sub(/^.*\//, "", base)
            sub(/\.ckpt$/, "", base)
            n = split(base, parts, "-")
            val_loss = parts[n] + 0
            if (!seen || val_loss < best_val_loss) {
                seen = 1
                best_val_loss = val_loss
                best_path = $0
            }
        }
        END {
            if (seen) {
                print best_path
            }
        }
    '
}

resolve_mae_checkpoint() {
    local source="$1"

    if [[ -f "${source}" ]]; then
        printf '%s\n' "${source}"
        return 0
    fi

    if [[ ! -d "${source}" ]]; then
        return 1
    fi

    find "${source}" -type f -name 'best-val-*.ckpt' | select_best_val_checkpoint
}

find_default_mae_checkpoint() {
    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs/ \
        -path '*/vit_mae_pretrain/*/autocast/*/checkpoints/best-val-*.ckpt' \
        | select_best_val_checkpoint
}

if [[ -z "${MAE_SOURCE}" ]]; then
    MAE_SOURCE="auto:outputs/*/vit_mae_pretrain/*/autocast/*/checkpoints/best-val-*.ckpt"
    MAE_CHECKPOINT="$(find_default_mae_checkpoint)"
elif ! MAE_CHECKPOINT="$(resolve_mae_checkpoint "${MAE_SOURCE}")"; then
    echo "FATAL: MAE source does not exist: ${MAE_SOURCE}" >&2
    exit 1
fi

if [[ -z "${MAE_CHECKPOINT}" ]]; then
    echo "FATAL: no best-val-*.ckpt found for MAE source: ${MAE_SOURCE}" >&2
    exit 1
fi

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
)

CRPS_BUDGET_HOURS="${CRPS_BUDGET_HOURS:-6}"
CRPS_LEARNING_RATE="${CRPS_LEARNING_RATE:-1e-4}"
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_vit_mae_to_crps"
N_MEMBERS=8
BS_PER_GPU=32

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="vit_mae_to_crps_${datamodule}_m${N_MEMBERS}"

    echo "Submitting MAE-initialized CRPS timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  mae source: ${MAE_SOURCE}"
    echo "  mae checkpoint: ${MAE_CHECKPOINT}"
    echo "  n_members: ${N_MEMBERS}"
    echo "  bs_per_gpu: ${BS_PER_GPU}"
    echo "  learning_rate: ${CRPS_LEARNING_RATE}"
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
        optimizer.learning_rate="${CRPS_LEARNING_RATE}" \
        +resume_from_checkpoint="${MAE_CHECKPOINT}" \
        +resume_weights_only=true \
        logging.wandb.log_model=false

    echo ""
    echo "---"
    echo ""
done

echo "All MAE-initialized CRPS timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/vit_mae_to_crps_*/retrieve.sh; do bash \"\$f\"; done"
