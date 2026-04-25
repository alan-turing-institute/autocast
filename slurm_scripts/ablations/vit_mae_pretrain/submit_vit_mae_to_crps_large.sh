#!/bin/bash

set -euo pipefail
# Short MAE-initialized CRPS fine-tuning run on CNS.
#
# Usage:
#   MAE_CHECKPOINT=/path/to/mae/encoder_processor_decoder.ckpt \
#     bash slurm_scripts/ablations/vit_mae_pretrain/submit_vit_mae_to_crps_large.sh
#
# Run submit_vit_mae_to_crps_timing.sh first. If COSINE_EPOCHS_BY_DATASET is
# left blank, this script derives max_epochs from the newest matching timing.ckpt.

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

declare -A COSINE_EPOCHS_BY_DATASET=(
    # ["conditioned_navier_stokes"]=...
)

CRPS_BUDGET_HOURS="${CRPS_BUDGET_HOURS:-4}"
if ! [[ "${CRPS_BUDGET_HOURS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FATAL: CRPS_BUDGET_HOURS must be a positive integer number of hours" >&2
    exit 1
fi

BUDGET_MAX_TIME="$(printf "00:%02d:59:00" "$((CRPS_BUDGET_HOURS - 1))")"
TIMEOUT_MIN=$((CRPS_BUDGET_HOURS * 60 - 1))
RUN_DRY_STATES=("true" "false")
RUN_GROUP="$(date +%Y-%m-%d)/vit_mae_to_crps"
N_MEMBERS=16
BS_PER_GPU=16

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/timing_vit_mae_to_crps/${run_id}/timing.ckpt" | sort | tail -n 1
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
        echo "  mae checkpoint: ${MAE_CHECKPOINT}"
        echo "  n_members: ${N_MEMBERS}"
        echo "  bs_per_gpu: ${BS_PER_GPU}"
        echo "  budget: ${CRPS_BUDGET_HOURS}h"
        echo "  cosine_epochs: ${cosine_epochs}"
        echo "  wandb.name: ${wandb_name}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            model.n_members="${N_MEMBERS}" \
            datamodule.batch_size="${BS_PER_GPU}" \
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
