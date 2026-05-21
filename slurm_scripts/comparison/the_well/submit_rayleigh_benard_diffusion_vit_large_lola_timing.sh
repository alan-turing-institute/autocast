#!/bin/bash

set -euo pipefail

# Submit a timing run for the unmasked LoLA-core Rayleigh-Benard diffusion
# ViT-large ablation. This uses LoLA's fixed epoch units:
#   train epoch_size 16,384 / global batch 256 = 64 train batches/epoch
#   valid epoch_size 4,096 / global batch 256 = 16 validation batches/epoch
#
# The default 20 epochs gives a longer warm/stable timing sample than the
# older 5-epoch timing scripts.
#
# Use the generated timing.ckpt to derive a 24h epoch budget:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/diffusion_vit_large_lola"
RUN_ID="${RUN_ID:-rb_diffusion_vit_large_lola_b256}"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing}"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-20}"
LOLA_TRAIN_EPOCH_SIZE="${LOLA_TRAIN_EPOCH_SIZE:-16384}"
LOLA_VALID_EPOCH_SIZE="${LOLA_VALID_EPOCH_SIZE:-4096}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-64}"
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

GLOBAL_BATCH_SIZE="$((PER_GPU_BATCH_SIZE * NUM_GPUS))"

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
TOTAL_TIMING_STEPS="$((NUM_TIMING_EPOCHS * TRAIN_BATCHES_PER_EPOCH))"

echo "Submitting Rayleigh-Benard LoLA-core diffusion timing run"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${CACHE_DIR}"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  train batches/epoch: ${TRAIN_BATCHES_PER_EPOCH}"
echo "  val batches/epoch: ${VAL_BATCHES_PER_EPOCH}"
echo "  total timing optimizer steps: ${TOTAL_TIMING_STEPS}"
echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
echo "  effective global batch: ${GLOBAL_BATCH_SIZE}"
echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"

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
    "+trainer.limit_train_batches=${TRAIN_BATCHES_PER_EPOCH}" \
    "+trainer.limit_val_batches=${VAL_BATCHES_PER_EPOCH}" \
    "+trainer.check_val_every_n_epoch=1" \
    "+trainer.num_sanity_val_steps=0" \
    "+trainer.enable_progress_bar=false" \
    "trainer.gradient_clip_val=1.0" \
    "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}"

echo ""
echo "Once the SLURM job completes, collect the timing result with:"
echo "  bash outputs/${RUN_GROUP}/${RUN_ID}/retrieve.sh"
