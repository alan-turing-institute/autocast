#!/bin/bash

set -euo pipefail

# Timing run for the fresh Rayleigh-Benard effective-batch 24h comparison:
# CRPS in cached LoLA latent space.
#
# Effective global batch = 32/GPU * 8 members * 4 GPUs = 1024.
# Each timing epoch is capped to 64 train batches so the epoch-stepped cosine
# schedule has useful granularity inside a 24h wall-clock budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_latent"
RUN_ID="rb_eff24_crps_latent_b32_m8"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing/rb_effective_batch_24h}"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
EFFECTIVE_BATCHES_PER_EPOCH="${EFFECTIVE_BATCHES_PER_EPOCH:-64}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
N_MEMBERS="${N_MEMBERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
NUM_GPUS=4

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

EFFECTIVE_GLOBAL_BATCH="$((PER_GPU_BATCH_SIZE * N_MEMBERS * NUM_GPUS))"
EFFECTIVE_UNITS_PER_EPOCH="$((EFFECTIVE_GLOBAL_BATCH * EFFECTIVE_BATCHES_PER_EPOCH))"

echo "Submitting RB effective-batch timing run"
echo "  method: CRPS latent"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${CACHE_DIR}"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
echo "  model.n_members: ${N_MEMBERS}"
echo "  effective global batch: ${EFFECTIVE_GLOBAL_BATCH}"
echo "  raw train batches/epoch: ${EFFECTIVE_BATCHES_PER_EPOCH}"
echo "  effective units/epoch: ${EFFECTIVE_UNITS_PER_EPOCH}"

uv run autocast time-epochs --kind processor --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    "local_experiment=${EXPERIMENT}" \
    "datamodule.data_path=${CACHE_DIR}" \
    "datamodule.batch_size=${PER_GPU_BATCH_SIZE}" \
    "datamodule.num_workers=${DATALOADER_NUM_WORKERS}" \
    "datamodule.pin_memory=true" \
    "datamodule.persistent_workers=true" \
    "datamodule.prefetch_factor=2" \
    "model.n_members=${N_MEMBERS}" \
    "+trainer.accumulate_grad_batches=1" \
    "+trainer.limit_train_batches=${EFFECTIVE_BATCHES_PER_EPOCH}" \
    "+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
    "+trainer.num_sanity_val_steps=0" \
    "+trainer.enable_progress_bar=false" \
    "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}"
