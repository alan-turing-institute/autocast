#!/bin/bash

set -euo pipefail

# Rayleigh-Benard latent masked-window flow-matching ViT-large run.
# Model: flow_matching_masked_window_vit with LoLA vit_large dimensions
# (hid_channels=1024, hid_blocks=16, attention_heads=8, patch_size=1,
# dropout=0.05, flow_ode_steps=50). Optimizer: adamw_half (LR=1e-4,
# warmup=0). Batch: 64/GPU on 4 GPUs = 256 global.
#
# This intentionally keeps the simple LoLA-style 4,096-epoch schedule for now
# rather than the optimizer-step-based conversion used in the baseline
# lola4096 submitter. We can revisit the step-budget optimization separately.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/fm_vit_large"
EXPERIMENT_NAME="the_well_rayleigh_benard_fm_vit_large_masked_window_lola4096"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
GIT_SHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
RUN_ID="${RUN_ID:-diff_rayleigh_benard_flow_matching_masked_window_vit_lola4096_${GIT_SHA}_$(date +%H%M%S)}"
LOLA_EPOCHS="${LOLA_EPOCHS:-4096}"
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

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting Rayleigh-Benard masked-window FM ViT-large training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  experiment_name: ${EXPERIMENT_NAME}"
    echo "  run_id: ${RUN_ID}"
    echo "  cache dir: ${CACHE_DIR}"
    echo "  LoLA-style epochs: ${LOLA_EPOCHS}"
    echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
    echo "  effective global batch: ${GLOBAL_BATCH_SIZE}"
    echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
    echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"
    echo "  trainer.log_every_n_steps: ${LOG_EVERY_N_STEPS}"
    echo "  trainer.max_time: ${BUDGET_MAX_TIME}"

    uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
        --run-id "${RUN_ID}" \
        local_experiment="${EXPERIMENT}" \
        processor@model.processor=flow_matching_masked_window_vit \
        backbone@model.processor.backbone=vit \
        experiment_name="${EXPERIMENT_NAME}" \
        datamodule.data_path="${CACHE_DIR}" \
        datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
        datamodule.num_workers="${DATALOADER_NUM_WORKERS}" \
        datamodule.pin_memory=true \
        datamodule.persistent_workers=true \
        datamodule.prefetch_factor=2 \
        logging.wandb.enabled=false \
        optimizer.scheduler_interval=epoch \
        optimizer.cosine_epochs="${LOLA_EPOCHS}" \
        hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        trainer.log_every_n_steps="${LOG_EVERY_N_STEPS}" \
        +trainer.max_epochs="${LOLA_EPOCHS}" \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
        # +trainer.enable_progress_bar=false \
        # +trainer.num_sanity_val_steps=0 \
        # logging.wandb.enabled=true \
done
