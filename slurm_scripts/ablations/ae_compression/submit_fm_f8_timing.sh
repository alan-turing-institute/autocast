#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../comparison/cached_latents/validate_cached_latents_against_ae.sh"
# CNS-only AE-compression ablation: 5-epoch FM-in-latent timing run on the
# f8 (8x8x8) cached latents.
# Re-times because the main-study cosine_epochs=3223 was derived on the
# 16x16x8 latent — the 8x8x8 latent has 4x fewer tokens per sample so the
# per-epoch wall is meaningfully different.
# Reuses processor/conditioned_navier_stokes/fm_vit_large.yaml (vit
# backbone, patch_size=1) — the smaller spatial grid is handled
# automatically by the ViT.

EXPERIMENT="processor/conditioned_navier_stokes/fm_vit_large"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-26/ae_cns64_de1b4b7_e1059d7"
DATAMODULE="conditioned_navier_stokes"

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

cache_dir="${AE_RUN_DIR}/cached_latents"

if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
    echo "Cache missing train/valid/test under ${cache_dir}" >&2
    exit 1
fi
if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
    echo "Cached-latents config mismatch vs AE training config" >&2
    exit 1
fi

echo "Submitting f8 FM-in-latent timing run"
echo "  datamodule: ${DATAMODULE}"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${cache_dir}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo ""

uv run autocast time-epochs --kind processor --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "fm_f8_b256_${DATAMODULE}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}" \
    datamodule.data_path="${cache_dir}"

echo ""
echo "Once SLURM job completes, collect with:"
echo "  bash outputs/${RUN_GROUP}/fm_f8_b256_${DATAMODULE}/retrieve.sh"
echo "Then derive COSINE_EPOCHS via:"
echo "  uv run autocast time-epochs --from-checkpoint outputs/${RUN_GROUP}/fm_f8_b256_${DATAMODULE}/timing.ckpt -b 24"
