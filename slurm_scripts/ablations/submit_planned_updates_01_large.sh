#!/bin/bash

set -euo pipefail

# Final 24h jobs for planned updates batch 01.
#
# Run submit_planned_updates_01_timing.sh first. This script derives each
# dataset's trainer.max_epochs from the newest matching timing checkpoint
# unless a value has been pinned in COSINE_EPOCHS_BY_DATASET.

declare -A COSINE_EPOCHS_BY_DATASET=(
    # Populate after timing, if desired:
    # ["gray_scott"]=...
    # ["gpe_laser_only_wake"]=...
    # ["conditioned_navier_stokes"]=...
    # ["advection_diffusion"]=...
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
TRAINING_SEED="${TRAINING_SEED:-42}"
RUN_DRY_STATES=("true" "false")
RUN_GROUP="$(date +%Y-%m-%d)/planned_updates_01"

declare -A EXPERIMENTS=(
    ["gray_scott"]="ablations/mc_dropout/gray_scott/crps_vit_azula_mc_dropout_large"
    ["gpe_laser_only_wake"]="ablations/mc_dropout/gpe_laser_wake_only/crps_vit_azula_mc_dropout_large"
    ["conditioned_navier_stokes"]="ablations/mc_dropout/conditioned_navier_stokes/crps_vit_azula_mc_dropout_large"
    ["advection_diffusion"]="ablations/mc_dropout/advection_diffusion/crps_vit_azula_mc_dropout_large"
)

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/timing_planned_updates_01/${run_id}/timing.ckpt" \
        | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs --from-checkpoint "${timing_ckpt}" -b 24 -m 0.02
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

    local run_id="mc_dropout_crps_${datamodule}"
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
        echo "Skipping ${datamodule}: could not parse trainer.max_epochs" >&2
        continue
    fi

    run_id="mc_dropout_crps_${datamodule}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting planned updates batch 01 training"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${cosine_epochs}"
        echo "  seed: ${TRAINING_SEED}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            seed="${TRAINING_SEED}" \
            logging.wandb.enabled=true \
            logging.wandb.name="${run_id}" \
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
