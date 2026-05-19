#!/bin/bash

set -euo pipefail

# Timing run for the fresh Rayleigh-Benard effective-batch 24h comparison:
# CRPS in ambient space via frozen LoLA encoder/decoder.
#
# Effective global batch = 32/GPU * 8 members * 4 GPUs = 1024.
# Each timing epoch is capped to 64 train batches so the epoch-stepped cosine
# schedule has useful granularity inside a 24h wall-clock budget.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_ambient"
RUN_ID="rb_eff24_crps_ambient_b32_m8"
RAW_DATA_DIR="${DATASETS_ROOT}/rayleigh_benard/data"
LOLA_AE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing/rb_effective_batch_24h}"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
EFFECTIVE_BATCHES_PER_EPOCH="${EFFECTIVE_BATCHES_PER_EPOCH:-64}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-8}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-32}"
N_MEMBERS="${N_MEMBERS:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
ENCODER_CHUNK_SIZE="${ENCODER_CHUNK_SIZE:-8}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-4}"
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

if [[ ! -f "${LOLA_AE_DIR}/config.yaml" ]] || [[ ! -f "${LOLA_AE_DIR}/state.pth" ]]; then
    echo "Missing LoLA config.yaml or state.pth under ${LOLA_AE_DIR}" >&2
    exit 1
fi

EFFECTIVE_GLOBAL_BATCH="$((PER_GPU_BATCH_SIZE * N_MEMBERS * NUM_GPUS))"
EFFECTIVE_UNITS_PER_EPOCH="$((EFFECTIVE_GLOBAL_BATCH * EFFECTIVE_BATCHES_PER_EPOCH))"

echo "Submitting RB effective-batch timing run"
echo "  method: CRPS ambient"
echo "  local_experiment: ${EXPERIMENT}"
echo "  raw data dir: ${RAW_DATA_DIR}"
echo "  LoLA AE dir: ${LOLA_AE_DIR}"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
echo "  model.n_members: ${N_MEMBERS}"
echo "  effective global batch: ${EFFECTIVE_GLOBAL_BATCH}"
echo "  raw train batches/epoch: ${EFFECTIVE_BATCHES_PER_EPOCH}"
echo "  effective units/epoch: ${EFFECTIVE_UNITS_PER_EPOCH}"
echo "  encoder chunk size: ${ENCODER_CHUNK_SIZE}"
echo "  decoder chunk size: ${DECODER_CHUNK_SIZE}"

uv run autocast time-epochs --kind epd --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    "local_experiment=${EXPERIMENT}" \
    "datamodule.well_base_path=${DATASETS_ROOT}" \
    "datamodule.batch_size=${PER_GPU_BATCH_SIZE}" \
    "datamodule.num_workers=${DATALOADER_NUM_WORKERS}" \
    "model.n_members=${N_MEMBERS}" \
    "model.encoder.runpath=${LOLA_AE_DIR}" \
    "model.encoder.chunk_size=${ENCODER_CHUNK_SIZE}" \
    "model.decoder.runpath=${LOLA_AE_DIR}" \
    "model.decoder.chunk_size=${DECODER_CHUNK_SIZE}" \
    "+trainer.accumulate_grad_batches=1" \
    "+trainer.limit_train_batches=${EFFECTIVE_BATCHES_PER_EPOCH}" \
    "+trainer.check_val_every_n_epoch=${CHECK_VAL_EVERY_N_EPOCH}" \
    "+trainer.num_sanity_val_steps=0" \
    "+trainer.enable_progress_bar=false" \
    "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}"
