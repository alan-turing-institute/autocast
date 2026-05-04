#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../comparison/cached_latents/validate_cached_latents_against_ae.sh"

# Quick sanity check: GPE m=16 latent CRPS — does the loss drop?
#
# Not a timing run, not a production run. Short max_time, modest cosine
# schedule so the LR still anneals, default callbacks, wandb on so the loss
# trajectory is visible live. Expected outcome: train_loss / val_loss visibly
# decrease within a handful of epochs.

CHECK_MAX_TIME="00:02:00:00"
TIMEOUT_MIN=125          # 2h05 — gives Lightning a few minutes to wrap up
COSINE_EPOCHS=160        # Choose around half of 326 for m=8
RUN_GROUP="$(date +%Y-%m-%d)/check_latent_crps_m16_gpe"

AE_RUN_DIR="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f"
CACHE_DIR="${AE_RUN_DIR}/cached_latents"
LOCAL_EXPERIMENT="processor/gpe_laser_wake_only/crps_vit_azula_large"
RUN_ID="latent_crps_m16_gpe_check"
RUN_DRY_STATES=("true" "false")

if [[ ! -d "${CACHE_DIR}/train" ]] || [[ ! -d "${CACHE_DIR}/valid" ]] || [[ ! -d "${CACHE_DIR}/test" ]]; then
    echo "FATAL: cache missing train/valid/test under ${CACHE_DIR}" >&2
    exit 1
fi
if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
    echo "FATAL: cached-latents config mismatch vs AE training config" >&2
    exit 1
fi

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting GPE m=16 latent CRPS sanity check"
    echo "  mode: ${run_label}"
    echo "  run_id: ${RUN_ID}"
    echo "  local_experiment: ${LOCAL_EXPERIMENT}"
    echo "  cache_dir: ${CACHE_DIR}"
    echo "  cosine_epochs: ${COSINE_EPOCHS}"
    echo "  max_time: ${CHECK_MAX_TIME}"

    uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
        --run-group "${RUN_GROUP}" \
        local_experiment="${LOCAL_EXPERIMENT}" \
        datamodule=cached_latents \
        datamodule.data_path="${CACHE_DIR}" \
        datamodule.batch_size=16 \
        model.n_members=16 \
        logging.wandb.enabled=true \
        logging.wandb.name="${RUN_ID}" \
        optimizer.cosine_epochs="${COSINE_EPOCHS}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${CHECK_MAX_TIME}" \
        +trainer.max_epochs="${COSINE_EPOCHS}"
done
