#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

MODE="${1:-final}"
if [[ "${MODE}" != "final" && "${MODE}" != "timing" ]]; then
    echo "Usage: $0 [timing|final] [all|crps_lola_pixel_ambient crps_ambient crps_latent fm_latent ...]" >&2
    exit 1
fi
shift || true

if (($# == 0)) || [[ "${1:-}" == "all" ]]; then
    RUN_KEYS=(crps_lola_pixel_ambient crps_ambient crps_latent fm_latent)
else
    RUN_KEYS=("$@")
fi

BUDGET_HOURS="${BUDGET_HOURS:-24}"
BUDGET_MARGIN="${BUDGET_MARGIN:-0.02}"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
EFFECTIVE_BATCHES_PER_EPOCH="${EFFECTIVE_BATCHES_PER_EPOCH:-64}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
TIMING_EPOCHS="${TIMING_EPOCHS:-3}"
NUM_GPUS="${NUM_GPUS:-4}"

set_run_defaults() {
    local key="$1"

    EXTRA_OVERRIDES=()
    USE_CACHE_DIR=false

    case "${key}" in
        crps_lola_pixel_ambient)
            METHOD_LABEL="CRPS ambient, LOLA pixel ViT"
            EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_lola_pixel_ambient"
            EXPERIMENT_NAME="the_well_rayleigh_benard_24h_crps_lola_pixel_ambient_b32m8"
            RUN_ID="rb_24h_crps_lola_pixel_ambient_b32m8"
            PER_GPU_BATCH_SIZE_DEFAULT=32
            N_MEMBERS=8
            DEFAULT_COSINE_EPOCHS=272
            ;;
        crps_ambient)
            METHOD_LABEL="CRPS ambient"
            EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_ambient"
            EXPERIMENT_NAME="the_well_rayleigh_benard_24h_crps_ambient_b32m8"
            RUN_ID="rb_24h_crps_ambient_b32m8"
            PER_GPU_BATCH_SIZE_DEFAULT=32
            N_MEMBERS=8
            DEFAULT_COSINE_EPOCHS=1642
            ;;
        crps_latent)
            METHOD_LABEL="CRPS latent"
            EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_latent"
            EXPERIMENT_NAME="the_well_rayleigh_benard_24h_crps_latent_b32m8"
            RUN_ID="rb_24h_crps_latent_b32m8"
            PER_GPU_BATCH_SIZE_DEFAULT=32
            N_MEMBERS=8
            DEFAULT_COSINE_EPOCHS=2299
            USE_CACHE_DIR=true
            ;;
        fm_latent)
            METHOD_LABEL="FM latent"
            EXPERIMENT="the_well/rayleigh_benard/fm_vit_large"
            EXPERIMENT_NAME="the_well_rayleigh_benard_24h_fm_latent_b256"
            RUN_ID="rb_24h_fm_latent_b256"
            PER_GPU_BATCH_SIZE_DEFAULT=256
            N_MEMBERS=1
            DEFAULT_COSINE_EPOCHS=2130
            USE_CACHE_DIR=true
            ;;
        *)
            echo "Unknown 24h run key: ${key}" >&2
            exit 1
            ;;
    esac
}

resolve_cosine_epochs() {
    local key="$1"
    local env_name="COSINE_EPOCHS_${key^^}"
    local specific_value="${!env_name:-}"

    if [[ -n "${specific_value}" ]]; then
        printf '%s\n' "${specific_value}"
    elif [[ -n "${COSINE_EPOCHS:-}" ]]; then
        printf '%s\n' "${COSINE_EPOCHS}"
    else
        printf '%s\n' "${DEFAULT_COSINE_EPOCHS}"
    fi
}

submit_one_run() {
    local key="$1"
    local run_dry="$2"
    local mode_label
    local per_gpu_batch_size
    local effective_global_batch
    local effective_units_per_epoch
    local cosine_epochs

    set_run_defaults "${key}"
    if [[ "${USE_CACHE_DIR}" == "true" ]]; then
        rb_require_cache
    fi

    mode_label="$(rb_print_mode "${run_dry}")"
    per_gpu_batch_size="${PER_GPU_BATCH_SIZE:-${PER_GPU_BATCH_SIZE_DEFAULT}}"
    effective_global_batch="$((per_gpu_batch_size * N_MEMBERS * NUM_GPUS))"
    effective_units_per_epoch="$((effective_global_batch * EFFECTIVE_BATCHES_PER_EPOCH))"

    COMMON_OVERRIDES=(
        "local_experiment=${EXPERIMENT}"
        "experiment_name=${EXPERIMENT_NAME}"
        "datamodule.batch_size=${per_gpu_batch_size}"
        "datamodule.num_workers=${DATALOADER_NUM_WORKERS}"
        "datamodule.pin_memory=true"
        "datamodule.persistent_workers=true"
        "datamodule.prefetch_factor=2"
        "logging.wandb.enabled=true"
        "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}"
        "hydra.launcher.timeout_min=${TIMEOUT_MIN}"
    )
    if [[ "${USE_CACHE_DIR}" == "true" ]]; then
        COMMON_OVERRIDES+=("datamodule.data_path=${RB_CACHE_DIR}")
    fi

    echo "Submitting RB 24h comparison ${MODE} run"
    echo "  mode: ${mode_label}"
    echo "  method: ${METHOD_LABEL}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  datamodule.batch_size: ${per_gpu_batch_size} per GPU"
    echo "  effective global batch: ${effective_global_batch}"
    echo "  effective units/epoch: ${effective_units_per_epoch}"

    if [[ "${MODE}" == "timing" ]]; then
        rb_submit time-epochs "${run_dry}" \
            --kind processor \
            --run-group "timing/rb_24h_comparison" \
            --run-id "${RUN_ID}" \
            -n "${TIMING_EPOCHS}" \
            -b "${BUDGET_HOURS}" \
            -m "${BUDGET_MARGIN}" \
            "${COMMON_OVERRIDES[@]}" \
            "+trainer.limit_train_batches=${EFFECTIVE_BATCHES_PER_EPOCH}" \
            "+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
            "+trainer.num_sanity_val_steps=0" \
            "+trainer.enable_progress_bar=false"
    else
        cosine_epochs="$(resolve_cosine_epochs "${key}")"
        echo "  cosine_epochs: ${cosine_epochs}"
        echo "  trainer.max_time: ${BUDGET_MAX_TIME}"

        rb_submit processor "${run_dry}" \
            --run-id "${EXPERIMENT_NAME}" \
            "${COMMON_OVERRIDES[@]}" \
            "optimizer.cosine_epochs=${cosine_epochs}" \
            "trainer.max_time=${BUDGET_MAX_TIME}" \
            "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}" \
            "+trainer.accumulate_grad_batches=1" \
            "+trainer.max_epochs=${cosine_epochs}" \
            "+trainer.limit_train_batches=${EFFECTIVE_BATCHES_PER_EPOCH}" \
            "+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
            "+trainer.num_sanity_val_steps=0" \
            "+trainer.enable_progress_bar=false" \
            "trainer.callbacks.0.every_n_train_steps_fraction=0.05" \
            "+trainer.callbacks.0.every_n_epochs=0" \
            "trainer.callbacks.0.save_top_k=-1" \
            "trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\""
    fi
}

for key in "${RUN_KEYS[@]}"; do
    while IFS= read -r run_dry; do
        submit_one_run "${key}" "${run_dry}"
    done < <(rb_run_dry_states)
done
