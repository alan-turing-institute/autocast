#!/bin/bash

set -euo pipefail

# Submit a short timing run for Rayleigh-Benard CRPS in ambient space. The model
# uses the frozen LoLA DCAE around the Azula ViT processor and computes CRPS
# after decoding back to raw Rayleigh-Benard fields.
# Use the generated timing.ckpt to derive a 24h epoch budget:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/crps_vit_azula_large_ambient"
RAW_DATA_DIR="${DATASETS_ROOT}/rayleigh_benard/data"
LOLA_AE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-3}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing}"
RUN_ID="${RUN_ID:-rb_crps_ambient_b1_acc64}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-64}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
ENCODER_CHUNK_SIZE="${ENCODER_CHUNK_SIZE:-8}"
DECODER_CHUNK_SIZE="${DECODER_CHUNK_SIZE:-4}"

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

echo "Submitting Rayleigh-Benard CRPS ambient timing run"
echo "  local_experiment: ${EXPERIMENT}"
echo "  raw data dir: ${RAW_DATA_DIR}"
echo "  LoLA AE dir: ${LOLA_AE_DIR}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo "  datamodule.batch_size: ${PER_GPU_BATCH_SIZE} per GPU"
echo "  trainer.accumulate_grad_batches: ${ACCUMULATE_GRAD_BATCHES}"
echo "  encoder chunk size: ${ENCODER_CHUNK_SIZE}"
echo "  decoder chunk size: ${DECODER_CHUNK_SIZE}"
echo "  datamodule.num_workers: ${DATALOADER_NUM_WORKERS}"
echo "  hydra.launcher.cpus_per_task: ${CPUS_PER_TASK}"

uv run autocast time-epochs --kind epd --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}" \
    datamodule.well_base_path="${DATASETS_ROOT}" \
    datamodule.batch_size="${PER_GPU_BATCH_SIZE}" \
    datamodule.num_workers="${DATALOADER_NUM_WORKERS}" \
    trainer.accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES}" \
    model.encoder.runpath="${LOLA_AE_DIR}" \
    model.encoder.chunk_size="${ENCODER_CHUNK_SIZE}" \
    model.decoder.runpath="${LOLA_AE_DIR}" \
    model.decoder.chunk_size="${DECODER_CHUNK_SIZE}" \
    hydra.launcher.cpus_per_task="${CPUS_PER_TASK}"

echo ""
echo "Once the SLURM job completes, collect the timing result with:"
echo "  bash outputs/${RUN_GROUP}/${RUN_ID}/retrieve.sh"
