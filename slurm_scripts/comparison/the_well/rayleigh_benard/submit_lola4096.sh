#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if (($# == 0)) || [[ "${1:-}" == "all" ]]; then
    RUN_KEYS=(fm fm_masked diffusion)
else
    RUN_KEYS=("$@")
fi

MAX_EPOCHS="${MAX_EPOCHS:-4096}"
TRAIN_BATCHES_PER_EPOCH="${TRAIN_BATCHES_PER_EPOCH:-64}"
DIFFUSION_VAL_BATCHES_PER_EPOCH="${DIFFUSION_VAL_BATCHES_PER_EPOCH:-16}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
NUM_GPUS="${NUM_GPUS:-4}"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-64}"

set_run_defaults() {
    local key="$1"

    EXTRA_OVERRIDES=()

    case "${key}" in
        fm)
            METHOD_LABEL="FM latent"
            EXPERIMENT="the_well/rayleigh_benard/fm_vit_large"
            EXPERIMENT_NAME="the_well_rayleigh_benard_fm_vit_large_lola4096"
            ;;
        fm_masked)
            METHOD_LABEL="FM latent, masked window"
            EXPERIMENT="the_well/rayleigh_benard/fm_vit_large_masked_window"
            EXPERIMENT_NAME="the_well_rayleigh_benard_fm_vit_large_masked_window_lola4096"
            ;;
        diffusion)
            METHOD_LABEL="Diffusion latent"
            EXPERIMENT="the_well/rayleigh_benard/diffusion_vit_large_lola"
            EXPERIMENT_NAME="the_well_rayleigh_benard_diffusion_vit_large_lola4096"
            EXTRA_OVERRIDES=(
                "trainer.gradient_clip_val=1.0"
                "+trainer.limit_val_batches=${DIFFUSION_VAL_BATCHES_PER_EPOCH}"
                "+trainer.check_val_every_n_epoch=1"
            )
            ;;
        *)
            echo "Unknown LOLA-4096 run key: ${key}" >&2
            exit 1
            ;;
    esac
}

submit_one_run() {
    local key="$1"
    local run_dry="$2"
    local mode_label
    local global_batch_size
    local total_optimizer_steps
    local val_cadence_steps

    set_run_defaults "${key}"
    rb_require_cache

    mode_label="$(rb_print_mode "${run_dry}")"
    global_batch_size="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"
    total_optimizer_steps="$((MAX_EPOCHS * TRAIN_BATCHES_PER_EPOCH))"
    val_cadence_steps="$((CHECK_VAL_EVERY_N_EPOCH * TRAIN_BATCHES_PER_EPOCH))"

    echo "Submitting RB LOLA-equivalent 4096 run"
    echo "  mode: ${mode_label}"
    echo "  method: ${METHOD_LABEL}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  cache dir: ${RB_CACHE_DIR}"
    echo "  max_epochs: ${MAX_EPOCHS}"
    echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
    echo "  total optimizer steps: ${total_optimizer_steps}"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  effective global batch: ${global_batch_size}"
    if [[ "${key}" != "diffusion" ]]; then
        echo "  check_val_every_n_epoch: ${CHECK_VAL_EVERY_N_EPOCH}"
        echo "  validation cadence: every ${val_cadence_steps} train steps"
    fi

    COMMON_OVERRIDES=(
        "local_experiment=${EXPERIMENT}"
        "experiment_name=${EXPERIMENT_NAME}"
        "datamodule.data_path=${RB_CACHE_DIR}"
        "datamodule.batch_size=${PER_GPU_BATCH_SIZE}"
        "datamodule.num_workers=${DATALOADER_NUM_WORKERS}"
        "datamodule.pin_memory=true"
        "datamodule.persistent_workers=true"
        "datamodule.prefetch_factor=2"
        "logging.wandb.enabled=true"
        "optimizer.cosine_epochs=${MAX_EPOCHS}"
        "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}"
        "hydra.launcher.timeout_min=${TIMEOUT_MIN}"
        "trainer.max_time=${BUDGET_MAX_TIME}"
        "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}"
        "+trainer.max_epochs=${MAX_EPOCHS}"
        "+trainer.limit_train_batches=${TRAIN_BATCHES_PER_EPOCH}"
        "+trainer.enable_progress_bar=false"
        "+trainer.num_sanity_val_steps=0"
        "trainer.callbacks.0.every_n_train_steps_fraction=0.05"
        "+trainer.callbacks.0.every_n_epochs=0"
        "trainer.callbacks.0.save_top_k=-1"
        "trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\""
    )
    if [[ "${key}" != "diffusion" ]]; then
        COMMON_OVERRIDES+=("+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}")
    fi

    rb_submit processor "${run_dry}" \
        --run-id "${EXPERIMENT_NAME}" \
        "${COMMON_OVERRIDES[@]}" \
        "${EXTRA_OVERRIDES[@]}"
}

for key in "${RUN_KEYS[@]}"; do
    while IFS= read -r run_dry; do
        submit_one_run "${key}" "${run_dry}"
    done < <(rb_run_dry_states)
done
