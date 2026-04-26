#!/bin/bash

set -euo pipefail

# 24h production jobs for the planned batch 02 m=4 CRPS follow-up.
#
# Run submit_planned_02_m4_followup_timing.sh first. This script resolves each
# trainer.max_epochs from either COSINE_EPOCHS_BY_RUN or the latest matching
# timing.ckpt under outputs/*/timing_planned_02_m4_followup/<run_id>/timing.ckpt.

declare -A COSINE_EPOCHS_BY_RUN=(
    # Optional pinned values from:
    #   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
RUN_GROUP="$(date +%Y-%m-%d)/planned_02_m4_followup"
RUN_DRY_STATES=("true" "false")

# run_id|datamodule|local_experiment
RUNS=(
    "vit_m4_gpe|gpe_laser_only_wake|epd/gpe_laser_wake_only/crps_vit_azula_large"
    "vit_m4_ad|advection_diffusion|epd/advection_diffusion/crps_vit_azula_large"
)

run_overrides() {
    local datamodule="$1"

    printf '%s\n' \
        "datamodule=${datamodule}" \
        "model.n_members=4" \
        "datamodule.batch_size=64"
}

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs/ -path "*/timing_planned_02_m4_followup/${run_id}/timing.ckpt" \
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
    local run_id="$1"
    local cached="${COSINE_EPOCHS_BY_RUN[$run_id]:-}"

    if [[ -n "${cached}" ]]; then
        printf '%s\n' "${cached}"
        return 0
    fi

    local timing_ckpt
    timing_ckpt="$(find_timing_checkpoint "${run_id}")"
    if [[ -z "${timing_ckpt}" ]]; then
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id datamodule experiment <<< "${run_spec}"

    if ! cosine_epochs="$(resolve_cosine_epochs "${run_id}")"; then
        echo "Skipping ${run_id}: no timing-derived cosine_epochs available" >&2
        continue
    fi
    if [[ -z "${cosine_epochs}" ]]; then
        echo "Skipping ${run_id}: could not parse trainer.max_epochs from timing output" >&2
        continue
    fi

    mapfile -t overrides < <(run_overrides "${datamodule}")

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting planned batch 02 m=4 follow-up production run"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            local_experiment="${experiment}" \
            "${overrides[@]}" \
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
