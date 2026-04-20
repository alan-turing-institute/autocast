#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/validate_cached_latents_against_ae.sh"
# Ablation: submit CRPS-in-cached-latent timing run for
# conditioned_navier_stokes only.
# Model: AzulaViTProcessor / vit_azula_large (hidden_dim=568, n_layers=12,
# num_heads=8, patch_size=1, n_noise_channels=1024). Optimizer: adamw_half
# (LR=2e-4, warmup=0). Batch: 32/GPU x n_members=8 internal expansion;
# AlphaFairCRPSLoss.

EXPERIMENT="processor/conditioned_navier_stokes/crps_vit_azula_large"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
DATAMODULE="conditioned_navier_stokes"

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

cache_dir="${AE_RUN_DIR}/cached_latents"

if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
    echo "Skipping ${DATAMODULE}: cache missing train/valid/test under ${cache_dir}" >&2
    exit 1
fi
if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
    echo "Skipping ${DATAMODULE}: cached-latents config mismatch vs AE training config" >&2
    exit 1
fi

echo "Submitting CRPS-in-latent ablation timing run"
echo "  datamodule: ${DATAMODULE}"
echo "  local_experiment: ${EXPERIMENT}"
echo "  cache dir: ${cache_dir}"
echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
echo "  budget: ${BUDGET_HOURS}h"
echo "  run_group: ${RUN_GROUP}"
echo ""

uv run autocast time-epochs --kind processor --mode slurm \
    --run-group "${RUN_GROUP}" \
    --run-id "crps_latent_ablation_b32_${DATAMODULE}" \
    -n "${NUM_TIMING_EPOCHS}" \
    -b "${BUDGET_HOURS}" \
    local_experiment="${EXPERIMENT}" \
    datamodule.data_path="${cache_dir}"

echo ""
echo "Once SLURM job completes, collect results with:"
echo "  for f in outputs/${RUN_GROUP}/crps_latent_ablation_b32_${DATAMODULE}/retrieve.sh; do bash \"\$f\"; done"
