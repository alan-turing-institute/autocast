#!/bin/bash

set -euo pipefail

# Final 24h FM-in-latent Rayleigh-Benard run on the LoLa/MiniWell cache.
# Model: flow_matching_vit with LoLa vit_small dimensions
# (hid_channels=512, hid_blocks=16, attention_heads=4, patch_size=1,
# dropout=0.05, flow_ode_steps=50). Optimizer: adamw_half (LR=1e-4,
# warmup=0). Batch: 256/GPU.
#
# COSINE_EPOCHS should come from a timing run so the cosine half-period fills
# the 24h budget. Pinned value below comes from:
#   seconds/epoch=60.6s, 24h budget, 2% margin -> max_epochs=1397
#
# This script uses, in order:
#   1. COSINE_EPOCHS=<int>
#   2. TIMING_CHECKPOINT=<path>/timing.ckpt
#   3. the latest outputs/*/rb_fm_vit_small_b256/timing.ckpt
#   4. PINNED_COSINE_EPOCHS=1397 from the timing output above

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/fm_vit_small"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
TIMING_RUN_ID="${TIMING_RUN_ID:-rb_fm_vit_small_b256}"
PINNED_COSINE_EPOCHS=1397
BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
RUN_DRY_STATES=("true" "false")

has_hdf5_split() {
    local split_dir="$1"
    compgen -G "${split_dir}/*.h5" > /dev/null || \
        compgen -G "${split_dir}/*.hdf5" > /dev/null
}

find_timing_checkpoint() {
    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/${TIMING_RUN_ID}/timing.ckpt" | sort | tail -n 1
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
    if [[ -n "${COSINE_EPOCHS:-}" ]]; then
        printf '%s\n' "${COSINE_EPOCHS}"
        return 0
    fi

    local timing_ckpt="${TIMING_CHECKPOINT:-}"
    if [[ -z "${timing_ckpt}" ]]; then
        timing_ckpt="$(find_timing_checkpoint)"
    fi

    if [[ -n "${timing_ckpt}" ]]; then
        derive_cosine_epochs_from_timing "${timing_ckpt}"
        return 0
    fi

    printf '%s\n' "${PINNED_COSINE_EPOCHS}"
}

for split in train valid test; do
    if ! has_hdf5_split "${CACHE_DIR}/${split}"; then
        echo "Missing ${split}/*.h5 or ${split}/*.hdf5 under ${CACHE_DIR}" >&2
        exit 1
    fi
done

if ! cosine_epochs="$(resolve_cosine_epochs)"; then
    exit 1
fi
if [[ -z "${cosine_epochs}" ]]; then
    echo "Unable to derive cosine epochs from timing result." >&2
    exit 1
fi
if ! [[ "${cosine_epochs}" =~ ^[0-9]+$ ]]; then
    echo "COSINE_EPOCHS must be a positive integer, got: ${cosine_epochs}" >&2
    exit 1
fi
if (( cosine_epochs < 1 )); then
    echo "COSINE_EPOCHS must be >= 1, got: ${cosine_epochs}" >&2
    exit 1
fi

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting Rayleigh-Benard FM ViT-small training"
    echo "  mode: ${run_label}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  cache dir: ${CACHE_DIR}"
    echo "  cosine_epochs: ${cosine_epochs}"
    echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
    echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"

    uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
        local_experiment="${EXPERIMENT}" \
        datamodule.data_path="${CACHE_DIR}" \
        datamodule.num_workers="${DATALOADER_NUM_WORKERS}" \
        datamodule.pin_memory=true \
        datamodule.persistent_workers=true \
        datamodule.prefetch_factor=2 \
        logging.wandb.enabled=true \
        optimizer.cosine_epochs="${cosine_epochs}" \
        hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        +trainer.max_epochs="${cosine_epochs}" \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
