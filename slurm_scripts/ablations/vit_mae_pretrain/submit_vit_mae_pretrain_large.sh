#!/bin/bash

set -euo pipefail
# 24h deterministic ViT MAE pretraining on CNS.
#
# Populate COSINE_EPOCHS_BY_DATASET after running
# submit_vit_mae_pretrain_timing.sh and extracting timing.ckpt with:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# If left blank, the script falls back to the newest matching timing.ckpt
# under outputs/*/timing_vit_mae_pretrain/.

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="ablations/vit_mae_pretrain/conditioned_navier_stokes/vit_azula_large_mae_no_ensemble"
)

declare -A COSINE_EPOCHS_BY_DATASET=(
    # ["conditioned_navier_stokes"]=...
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")
RUN_GROUP="$(date +%Y-%m-%d)/vit_mae_pretrain"

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/timing_vit_mae_pretrain/${run_id}/timing.ckpt" | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs --from-checkpoint "${timing_ckpt}" -b 24
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

    local run_id="vit_mae_pretrain_${datamodule}"
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

    wandb_name="vit_mae_pretrain_no_ensemble"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting ViT MAE pretraining"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${cosine_epochs}"
        echo "  wandb.name: ${wandb_name}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            logging.wandb.enabled=true \
            logging.wandb.name="${wandb_name}" \
            logging.wandb.log_model=all \
            optimizer.cosine_epochs="${cosine_epochs}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${cosine_epochs}" \
            trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
            trainer.callbacks.0.every_n_epochs=0 \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
    done
done
