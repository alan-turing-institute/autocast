#!/bin/bash

set -euo pipefail

# Submit a short timing run for Rayleigh-Benard CRPS in cached LoLA latent space.
# The run uses the same MiniWell cache as the FM ViT-large LoLA-4096 baseline.
# Use the generated timing.ckpt to derive a 24h epoch budget:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_latent"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing}"
RUN_ID="${RUN_ID:-rb_crps_latent_b32_acc2}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

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

echo "Submitting Rayleigh-Benard CRPS latent timing run"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${CACHE_DIR}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
echo "  trainer.accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"

uv run autocast time-epochs --kind processor --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}" \
    datamodule.data_path="${CACHE_DIR}" \
    datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
    datamodule.num_workers="${DATALOADER_NUM_WORKERS}" \
    datamodule.pin_memory=true \
    datamodule.persistent_workers=true \
    datamodule.prefetch_factor=2 \
    +trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
    hydra.launcher.cpus_per_task="${CPUS_PER_TASK}"

echo ""
echo "Once the SLURM job completes, collect the timing result with:"
echo "  bash outputs/${RUN_GROUP}/${RUN_ID}/retrieve.sh"
