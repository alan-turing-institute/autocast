#!/bin/bash

set -euo pipefail

# Final 24h run for the fresh Rayleigh-Benard effective-batch comparison:
# CRPS in ambient space with the LOLA pixel-space ViT hyperparameters.
#
# Effective global batch = 32/GPU * 8 members * 4 GPUs = 1024. Epochs are
# fixed-size budget epochs of 64 train batches by default, not full dataset
# passes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_lola_pixel_ambient"
EXPERIMENT_NAME="the_well_rayleigh_benard_effbatch24h_crps_ambient_lola_pixel_b32_m8"
RUN_ID="rb_eff24_crps_ambient_lola_pixel_b32_m8"
RAW_DATA_DIR="${DATASETS_ROOT}/rayleigh_benard/data"
TIMING_GROUP_GLOB="${TIMING_GROUP_GLOB:-*/timing/rb_effective_batch_24h}"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
BUDGET_MARGIN="${BUDGET_MARGIN:-0.02}"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
EFFECTIVE_BATCHES_PER_EPOCH="${EFFECTIVE_BATCHES_PER_EPOCH:-64}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
N_MEMBERS="${N_MEMBERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-false}"
if [[ "${DRY_RUN_ONLY}" == "true" ]]; then
    RUN_DRY_STATES=("true")
else
    RUN_DRY_STATES=("true" "false")
fi
NUM_GPUS=4

has_hdf5_split() {
    local split_dir="$1"
    compgen -G "${split_dir}/*.h5" > /dev/null || \
        compgen -G "${split_dir}/*.hdf5" > /dev/null
}

for split in train valid test; do
    if ! has_hdf5_split "${RAW_DATA_DIR}/${split}"; then
        echo "Missing ${split}/*.h5 or ${split}/*.hdf5 under ${RAW_DATA_DIR}" >&2
        exit 1
    fi
done

find_timing_checkpoint() {
    if [[ ! -d outputs ]]; then
        return 0
    fi
    find outputs -path "*/${TIMING_GROUP_GLOB}/${RUN_ID}/timing.ckpt" | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs \
            --from-checkpoint "${timing_ckpt}" \
            -b "${BUDGET_HOURS}" \
            -m "${BUDGET_MARGIN}"
    )"

    sed -n 's/.*trainer.max_epochs=\([0-9][0-9]*\).*/\1/p' <<< "${result}" | tail -n 1
}

resolve_cosine_epochs() {
    local timing_ckpt

    if [[ -n "${COSINE_EPOCHS_CRPS_AMBIENT_LOLA_PIXEL:-}" ]]; then
        printf '%s\n' "${COSINE_EPOCHS_CRPS_AMBIENT_LOLA_PIXEL}"
        return 0
    fi

    if [[ -n "${COSINE_EPOCHS:-}" ]]; then
        printf '%s\n' "${COSINE_EPOCHS}"
        return 0
    fi

    timing_ckpt="$(find_timing_checkpoint)"
    if [[ -z "${timing_ckpt}" ]]; then
        echo "No timing checkpoint found for ${RUN_ID}." >&2
        echo "Run ${SCRIPT_DIR}/submit_crps_ambient_lola_pixel_timing.sh first, or set COSINE_EPOCHS_CRPS_AMBIENT_LOLA_PIXEL=<epochs>." >&2
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

if ! COSINE_EPOCHS_RESOLVED="$(resolve_cosine_epochs)"; then
    exit 1
fi
if [[ -z "${COSINE_EPOCHS_RESOLVED}" ]] || ! [[ "${COSINE_EPOCHS_RESOLVED}" =~ ^[0-9]+$ ]] || (( COSINE_EPOCHS_RESOLVED < 1 )); then
    echo "Invalid cosine epoch count: ${COSINE_EPOCHS_RESOLVED}" >&2
    exit 1
fi

EFFECTIVE_GLOBAL_BATCH="$((PER_GPU_BATCH_SIZE * N_MEMBERS * NUM_GPUS))"
EFFECTIVE_UNITS_PER_EPOCH="$((EFFECTIVE_GLOBAL_BATCH * EFFECTIVE_BATCHES_PER_EPOCH))"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting RB effective-batch 24h run"
    echo "  mode: ${run_label}"
    echo "  method: CRPS ambient LOLA pixel ViT"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  raw data dir: ${RAW_DATA_DIR}"
    echo "  cosine_epochs: ${COSINE_EPOCHS_RESOLVED}"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  model.n_members: ${N_MEMBERS}"
    echo "  effective global batch: ${EFFECTIVE_GLOBAL_BATCH}"
    echo "  raw train batches/epoch: ${EFFECTIVE_BATCHES_PER_EPOCH}"
    echo "  effective units/epoch: ${EFFECTIVE_UNITS_PER_EPOCH}"
    echo "  check_val_every_n_epoch: ${CHECK_VAL_EVERY_N_EPOCH}"
    echo "  trainer.max_time: ${BUDGET_MAX_TIME}"

    uv run autocast epd --mode slurm --run-id "${EXPERIMENT_NAME}" "${dry_run_arg[@]}" \
        "local_experiment=${EXPERIMENT}" \
        "experiment_name=${EXPERIMENT_NAME}" \
        "datamodule.well_base_path=${DATASETS_ROOT}" \
        "datamodule.batch_size=${PER_GPU_BATCH_SIZE}" \
        "datamodule.num_workers=${DATALOADER_NUM_WORKERS}" \
        "model.n_members=${N_MEMBERS}" \
        "logging.wandb.enabled=true" \
        "optimizer.cosine_epochs=${COSINE_EPOCHS_RESOLVED}" \
        "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
        "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
        "trainer.max_time=${BUDGET_MAX_TIME}" \
        "trainer.log_every_n_steps=${LOG_EVERY_N_STEPS}" \
        "+trainer.accumulate_grad_batches=1" \
        "+trainer.max_epochs=${COSINE_EPOCHS_RESOLVED}" \
        "+trainer.limit_train_batches=${EFFECTIVE_BATCHES_PER_EPOCH}" \
        "+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
        "+trainer.num_sanity_val_steps=0" \
        "+trainer.enable_progress_bar=false" \
        "trainer.callbacks.0.every_n_train_steps_fraction=0.05" \
        "+trainer.callbacks.0.every_n_epochs=0" \
        "trainer.callbacks.0.save_top_k=-1" \
        "trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\""
done
