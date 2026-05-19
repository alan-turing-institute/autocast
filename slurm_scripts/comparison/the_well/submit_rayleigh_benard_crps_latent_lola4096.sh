#!/bin/bash

set -euo pipefail

# LoLA-style Rayleigh-Benard CRPS baseline in cached latent space.
# This matches the FM ViT-large LoLA-4096 sample budget:
#   4,096 epochs * 16,384 samples/epoch = 67,108,864 samples
# with global effective batch 256 and 64 optimizer steps per LoLA epoch.
# Since CRPS uses gradient accumulation, limit_train_batches counts raw
# per-rank batches, not optimizer steps.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_latent"
EXPERIMENT_NAME="the_well_rayleigh_benard_crps_latent_vit_azula_large_lola4096"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
MAX_EPOCHS="${MAX_EPOCHS:-4096}"
LOLA_EPOCH_SIZE="${LOLA_EPOCH_SIZE:-16384}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
NUM_GPUS=4
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"
EFFECTIVE_GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS * ACCUMULATE_GRAD_BATCHES))"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-64}"
RUN_DRY_STATES=("true" "false")

has_hdf5_split() {
    local split_dir="$1"
    compgen -G "${split_dir}/*.h5" > /dev/null || \
        compgen -G "${split_dir}/*.hdf5" > /dev/null
}

for split in train valid test; do
    if ! has_hdf5_split "${CACHE_DIR}/${split}"; then
        echo "Missing ${split}/*.h5 or ${split}/*.hdf5 under ${CACHE_DIR}" >&2
        exit 1
    fi
done

if (( LOLA_EPOCH_SIZE % GLOBAL_BATCH_SIZE != 0 )); then
    echo "LOLA_EPOCH_SIZE must divide global batch size." >&2
    echo "  LOLA_EPOCH_SIZE: ${LOLA_EPOCH_SIZE}" >&2
    echo "  global batch: ${GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

TRAIN_BATCHES_PER_EPOCH="$((LOLA_EPOCH_SIZE / GLOBAL_BATCH_SIZE))"

if (( TRAIN_BATCHES_PER_EPOCH % ACCUMULATE_GRAD_BATCHES != 0 )); then
    echo "train batches/epoch must divide accumulate_grad_batches." >&2
    echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}" >&2
    echo "  accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}" >&2
    echo "  effective global batch: ${EFFECTIVE_GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

OPTIMIZER_STEPS_PER_EPOCH="$((TRAIN_BATCHES_PER_EPOCH / ACCUMULATE_GRAD_BATCHES))"
TOTAL_OPTIMIZER_STEPS="$((MAX_EPOCHS * OPTIMIZER_STEPS_PER_EPOCH))"
VAL_CADENCE_STEPS="$((CHECK_VAL_EVERY_N_EPOCH * OPTIMIZER_STEPS_PER_EPOCH))"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting LoLA-style Rayleigh-Benard CRPS latent training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  cache dir: ${CACHE_DIR}"
    echo "  max_epochs: ${MAX_EPOCHS}"
    echo "  LoLA epoch size: ${LOLA_EPOCH_SIZE}"
    echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
    echo "  optimizer steps/epoch: ${OPTIMIZER_STEPS_PER_EPOCH}"
    echo "  total optimizer steps: ${TOTAL_OPTIMIZER_STEPS}"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  trainer.accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
    echo "  global batch: ${GLOBAL_BATCH_SIZE}"
    echo "  effective global batch: ${EFFECTIVE_GLOBAL_BATCH_SIZE}"
    echo "  check_val_every_n_epoch: ${CHECK_VAL_EVERY_N_EPOCH}"
    echo "  validation cadence: every ${VAL_CADENCE_STEPS} optimizer steps"
    echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
    echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"
    echo "  trainer.log_every_n_steps: ${LOG_EVERY_N_STEPS}"
    echo "  trainer.max_time: ${BUDGET_MAX_TIME}"

    uv run autocast processor --mode slurm --run-id "${EXPERIMENT_NAME}" "${dry_run_arg[@]}" \
        local_experiment="${EXPERIMENT}" \
        experiment_name="${EXPERIMENT_NAME}" \
        datamodule.data_path="${CACHE_DIR}" \
        datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
        datamodule.num_workers="${DATALOADER_NUM_WORKERS}" \
        datamodule.pin_memory=true \
        datamodule.persistent_workers=true \
        datamodule.prefetch_factor=2 \
        logging.wandb.enabled=true \
        optimizer.cosine_epochs="${MAX_EPOCHS}" \
        hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        +trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
        +trainer.max_epochs="${MAX_EPOCHS}" \
        +trainer.limit_train_batches="${TRAIN_BATCHES_PER_EPOCH}" \
        +trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
        +trainer.enable_progress_bar=false \
        +trainer.num_sanity_val_steps=0 \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
