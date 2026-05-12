#!/bin/bash

set -euo pipefail

# Submit a short timing run for the Rayleigh-Benard LoLa/MiniWell latent cache.
# Model: flow_matching_vit with LoLa vit_small dimensions
# (hid_channels=512, hid_blocks=16, attention_heads=4, patch_size=1,
# dropout=0.05, flow_ode_steps=50).
#
# Use the generated timing.ckpt to derive COSINE_EPOCHS for the 24h run:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS_ROOT="${AUTOCAST_DATASETS:-${REPO_ROOT}/datasets}"

EXPERIMENT="the_well/rayleigh_benard/fm_vit_small"
CACHE_DIR="${DATASETS_ROOT}/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard"
BUDGET_HOURS="${BUDGET_HOURS:-24}"
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing}"
RUN_ID="${RUN_ID:-rb_fm_vit_small_b256}"

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

echo "Submitting Rayleigh-Benard FM ViT-small timing run"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${CACHE_DIR}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"

uv run autocast time-epochs --kind processor --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}" \
    datamodule.data_path="${CACHE_DIR}"

echo ""
echo "Once the SLURM job completes, collect the timing result with:"
echo "  bash outputs/${RUN_GROUP}/${RUN_ID}/retrieve.sh"
