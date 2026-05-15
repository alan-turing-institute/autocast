#!/bin/bash

set -euo pipefail

# LoLA-style Rayleigh-Benard latent flow-matching ViT-large run.
# Model: flow_matching_vit with LoLA vit_large dimensions
# (hid_channels=1024, hid_blocks=16, attention_heads=8, patch_size=1,
# dropout=0.05, flow_ode_steps=50). Optimizer: adamw_half (LR=1e-4,
# warmup=0). Batch: 64/GPU on 4 GPUs = 256 global.
#
# LoLA defines an epoch as a fixed 16,384-sample training unit:
#   train epoch_size 16,384 / global batch 256 = 64 optimizer steps/epoch
#   4,096 epochs * 64 steps/epoch = 262,144 optimizer steps
#
# We reproduce that with Lightning's integer limit_train_batches=64. Validation
# remains a full validation pass, but only every 8 epochs (~512 train steps),
# which keeps validation compute near LoLA's 16 validation batches per epoch
# without repeatedly evaluating a deterministic prefix of the validation set.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/fm_vit_large"
EXPERIMENT_NAME="the_well_rayleigh_benard_fm_vit_large_lola4096"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
MAX_EPOCHS="${MAX_EPOCHS:-4096}"
TRAIN_BATCHES_PER_EPOCH="${TRAIN_BATCHES_PER_EPOCH:-64}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
NUM_GPUS=4
GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"
TOTAL_OPTIMIZER_STEPS="$((MAX_EPOCHS * TRAIN_BATCHES_PER_EPOCH))"
VAL_CADENCE_STEPS="$((CHECK_VAL_EVERY_N_EPOCH * TRAIN_BATCHES_PER_EPOCH))"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
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

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting LoLA-style Rayleigh-Benard FM ViT-large training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  cache dir: ${CACHE_DIR}"
    echo "  max_epochs: ${MAX_EPOCHS}"
    echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
    echo "  total optimizer steps: ${TOTAL_OPTIMIZER_STEPS}"
    echo "  check_val_every_n_epoch: ${CHECK_VAL_EVERY_N_EPOCH}"
    echo "  validation cadence: every ${VAL_CADENCE_STEPS} train steps"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  effective global batch: ${GLOBAL_BATCH_SIZE}"
    echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
    echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"
    echo "  trainer.max_time: ${BUDGET_MAX_TIME}"

    uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
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
        trainer.max_time="${BUDGET_MAX_TIME}" \
        +trainer.max_epochs="${MAX_EPOCHS}" \
        +trainer.limit_train_batches="${TRAIN_BATCHES_PER_EPOCH}" \
        +trainer.check_val_every_n_epoch="${CHECK_VAL_EVERY_N_EPOCH}" \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
