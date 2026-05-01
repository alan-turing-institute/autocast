#!/bin/bash

set -euo pipefail
# Short MAE-initialized CRPS fine-tuning run on CNS.
#
# Usage:
#   bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_large.sh
#   # or explicitly:
#   MAE_CHECKPOINT=/path/to/mae/run_dir \
#     bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_large.sh
#
# Run submit_vit_mae_to_crps_timing.sh first. If COSINE_EPOCHS_BY_DATASET is
# left blank, this script derives max_epochs from the newest matching timing.ckpt.

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

MAE_CHECKPOINT="$(realpath "${MAE_CHECKPOINT}")"

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
)

declare -A COSINE_EPOCHS_BY_DATASET=(
    # Pinned from outputs/2026-04-27/timing_vit_mae_to_crps/*/retrieve.sh at
    # 6h budget, 2% margin (uv run autocast time-epochs -b 6 -m 0.02).
    ["conditioned_navier_stokes"]=114
)

CRPS_BUDGET_HOURS="${CRPS_BUDGET_HOURS:-6}"
CRPS_LEARNING_RATE="${CRPS_LEARNING_RATE:-1e-4}"
if ! [[ "${CRPS_BUDGET_HOURS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: CRPS_BUDGET_HOURS must be a positive integer number of hours" >&2
    exit 1
fi

BUDGET_MAX_TIME="$(printf "00:%02d:59:00" "$((CRPS_BUDGET_HOURS - 1))")"
TIMEOUT_MIN=$((CRPS_BUDGET_HOURS * 60 - 1))
RUN_DRY_STATES=("true" "false")
RUN_GROUP="$(date +%Y-%m-%d)/vit_mae_to_crps"
N_MEMBERS=8
BS_PER_GPU=32

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs/ -path "*/timing_vit_mae_to_crps/${run_id}/timing.ckpt" | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs \
            --from-checkpoint "${timing_ckpt}" \
            -b "${CRPS_BUDGET_HOURS}" \
            -m 0.02
    )"

    sed -n 's/.*trainer.max_epochs=\([0-9][0-9]*\).*/\1/p' <<< "${result}" | tail -n 1
}

resolve_cosine_epochs() {
    local datamodule="$1"
    local cached="${COSINE_EPOCHS_BY_DATASET[$datamodule]:-}"

    if [[ -n "${cached}" ]]; then
        printf '%s\n' "${cached}"
        return 0
    fi

    local run_id="vit_mae_to_crps_${datamodule}_m${N_MEMBERS}"
    local timing_ckpt
    timing_ckpt="$(find_timing_checkpoint "${run_id}")"

    if [[ -z "${timing_ckpt}" ]]; then
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    if ! cosine_epochs="$(resolve_cosine_epochs "${datamodule}")"; then
        echo "Skipping ${datamodule}: no timing-derived cosine_epochs available" >&2
        continue
    fi
    if [[ -z "${cosine_epochs}" ]]; then
        echo "Skipping ${datamodule}: could not parse trainer.max_epochs from timing output" >&2
        continue
    fi

    wandb_name="vit_mae_to_crps_m${N_MEMBERS}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting MAE-initialized CRPS fine-tuning"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  mae source: ${MAE_SOURCE}"
        echo "  mae checkpoint: ${MAE_CHECKPOINT}"
        echo "  n_members: ${N_MEMBERS}"
        echo "  bs_per_gpu: ${BS_PER_GPU}"
        echo "  learning_rate: ${CRPS_LEARNING_RATE}"
        echo "  budget: ${CRPS_BUDGET_HOURS}h"
        echo "  cosine_epochs: ${cosine_epochs}"
        echo "  wandb.name: ${wandb_name}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            model.n_members="${N_MEMBERS}" \
            datamodule.batch_size="${BS_PER_GPU}" \
            optimizer.learning_rate="${CRPS_LEARNING_RATE}" \
            +resume_from_checkpoint="${MAE_CHECKPOINT}" \
            +resume_weights_only=true \
            logging.wandb.enabled=true \
            logging.wandb.name="${wandb_name}" \
            logging.wandb.log_model=false \
            optimizer.cosine_epochs="${cosine_epochs}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${cosine_epochs}" \
            trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
            +trainer.callbacks.0.every_n_epochs=0 \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
    done
done
