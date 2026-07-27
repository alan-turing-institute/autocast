#!/bin/bash

set -euo pipefail

# Final 24h job for planned updates batch 03.
#
# Run submit_planned_updates_03_timing.sh first. COSINE_EPOCHS may be supplied
# explicitly, otherwise this script derives it from the latest matching timing
# checkpoint.

EXPERIMENT="ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_fno_80m"
RUN_ID="fno_m8_crps_cns"
TIMING_GROUP="timing_planned_updates_03"
RUN_GROUP="$(date +%Y-%m-%d)/planned_updates_03"
BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
TRAINING_SEED="${TRAINING_SEED:-42}"
RUN_DRY_STATES=("true" "false")

find_timing_checkpoint() {
    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/${TIMING_GROUP}/${RUN_ID}/timing.ckpt" \
        | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs \
            --from-checkpoint "${timing_ckpt}" -b 24 -m 0.02
    )"

    sed -n 's/.*trainer.max_epochs=\([0-9][0-9]*\).*/\1/p' \
        <<< "${result}" | tail -n 1
}

resolve_cosine_epochs() {
    if [[ -n "${COSINE_EPOCHS:-}" ]]; then
        printf '%s\n' "${COSINE_EPOCHS}"
        return 0
    fi

    local timing_ckpt
    timing_ckpt="$(find_timing_checkpoint)"
    if [[ -z "${timing_ckpt}" ]]; then
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

if ! cosine_epochs="$(resolve_cosine_epochs)"; then
    echo "FATAL: no FNO timing checkpoint or COSINE_EPOCHS override found" >&2
    exit 1
fi
if [[ -z "${cosine_epochs}" ]]; then
    echo "FATAL: could not parse trainer.max_epochs from timing output" >&2
    exit 1
fi

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting planned updates batch 03 training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  cosine_epochs: ${cosine_epochs}"
    echo "  seed: ${TRAINING_SEED}"
    echo "  run_group: ${RUN_GROUP}"

    uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
        --run-group "${RUN_GROUP}" \
        local_experiment="${EXPERIMENT}" \
        seed="${TRAINING_SEED}" \
        logging.wandb.enabled=true \
        logging.wandb.name="${RUN_ID}" \
        optimizer.cosine_epochs="${cosine_epochs}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        +trainer.max_epochs="${cosine_epochs}" \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
