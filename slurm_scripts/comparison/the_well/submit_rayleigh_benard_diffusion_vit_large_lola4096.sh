#!/bin/bash

set -euo pipefail

# LoLA-style Rayleigh-Benard latent diffusion ViT-large run.
# Model: diffusion_vit with LoLA vit_large dimensions
# (hid_channels=1024, hid_blocks=16, attention_heads=8, patch_size=1,
# dropout=0.05, qk_norm=true, rope=true), LoLA log-logit schedule,
# LoLA EDM preconditioning, and 50 Euler sampler steps.
# Optimizer: adamw_half (LR=1e-4, warmup=0, grad clip=1).
#
# LoLA's train_diffusion.yaml defines fixed epoch sample counts:
#   train epoch_size 16,384 / global batch 256 = 64 optimizer steps/epoch
#   valid epoch_size 4,096 / global batch 256 = 16 validation batches/epoch
#   4,096 epochs * 64 steps/epoch = 262,144 optimizer steps
#
# We keep global batch 256 on this repo's 4-GPU Slurm preset by using
# 64 samples/GPU. LoLA used 8 GPUs with 32 samples/GPU; the optimizer-step
# and sample-count accounting is unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/diffusion_vit_large_lola"
EXPERIMENT_NAME="the_well_rayleigh_benard_diffusion_vit_large_lola4096"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
MAX_EPOCHS="${MAX_EPOCHS:-4096}"
LOLA_TRAIN_EPOCH_SIZE="${LOLA_TRAIN_EPOCH_SIZE:-16384}"
LOLA_VALID_EPOCH_SIZE="${LOLA_VALID_EPOCH_SIZE:-4096}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
NUM_GPUS=4
GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"
BUDGET_MAX_TIME="${BUDGET_MAX_TIME:-00:23:59:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
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

if (( LOLA_TRAIN_EPOCH_SIZE % GLOBAL_BATCH_SIZE != 0 )); then
    echo "LOLA_TRAIN_EPOCH_SIZE must divide global batch size." >&2
    echo "  LOLA_TRAIN_EPOCH_SIZE: ${LOLA_TRAIN_EPOCH_SIZE}" >&2
    echo "  global batch: ${GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

if (( LOLA_VALID_EPOCH_SIZE % GLOBAL_BATCH_SIZE != 0 )); then
    echo "LOLA_VALID_EPOCH_SIZE must divide global batch size." >&2
    echo "  LOLA_VALID_EPOCH_SIZE: ${LOLA_VALID_EPOCH_SIZE}" >&2
    echo "  global batch: ${GLOBAL_BATCH_SIZE}" >&2
    exit 1
fi

TRAIN_BATCHES_PER_EPOCH="$((LOLA_TRAIN_EPOCH_SIZE / GLOBAL_BATCH_SIZE))"
VAL_BATCHES_PER_EPOCH="$((LOLA_VALID_EPOCH_SIZE / GLOBAL_BATCH_SIZE))"
TOTAL_OPTIMIZER_STEPS="$((MAX_EPOCHS * TRAIN_BATCHES_PER_EPOCH))"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting LoLA-style Rayleigh-Benard diffusion ViT-large training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  cache dir: ${CACHE_DIR}"
    echo "  max_epochs: ${MAX_EPOCHS}"
    echo "  LoLA train epoch size: ${LOLA_TRAIN_EPOCH_SIZE}"
    echo "  LoLA valid epoch size: ${LOLA_VALID_EPOCH_SIZE}"
    echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
    echo "  val batches/epoch: ${VAL_BATCHES_PER_EPOCH}"
    echo "  total optimizer steps: ${TOTAL_OPTIMIZER_STEPS}"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  effective global batch: ${GLOBAL_BATCH_SIZE}"
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
        trainer.max_time="${BUDGET_MAX_TIME}" \
        trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
        trainer.gradient_clip_val=1.0 \
        +trainer.max_epochs="${MAX_EPOCHS}" \
        +trainer.limit_train_batches="${TRAIN_BATCHES_PER_EPOCH}" \
        +trainer.limit_val_batches="${VAL_BATCHES_PER_EPOCH}" \
        +trainer.check_val_every_n_epoch=1 \
        +trainer.enable_progress_bar=false \
        +trainer.num_sanity_val_steps=0 \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
