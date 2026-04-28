#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../comparison/cached_latents/validate_cached_latents_against_ae.sh"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# CNS-only AE-compression ablation: cache latents from the f8 (8x8x8) AE.
# Mirrors slurm_scripts/comparison/cached_latents/submit_cache_latents.sh
# but targets the single ae_dc_large_f8 run rather than the 4-dataset
# main-study AEs.
#
# The cache_latents experiment yaml supplies the encoder/decoder shape via
# the dc_large overrides — those load defaults that the AE checkpoint then
# overrides at load time, so the same yaml works for both 3-level and
# 4-level AEs without modification.

DATAMODULE="conditioned_navier_stokes"
EXPERIMENT="cache_latents/conditioned_navier_stokes/cache_latents"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-26/ae_cns64_de1b4b7_e1059d7"

ckpt="${AE_RUN_DIR}/autoencoder.ckpt"
if [[ ! -f "$ckpt" ]]; then
    ckpt="$(ls -t "${AE_RUN_DIR}"/autocast/*/checkpoints/latest-*.ckpt 2>/dev/null | head -n 1 || true)"
    if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
        echo "No autoencoder.ckpt or latest-*.ckpt under ${AE_RUN_DIR}" >&2
        exit 1
    fi
    echo "Using temp checkpoint (AE still training?): ${ckpt}" >&2
fi

cache_workdir="${AE_RUN_DIR}/cached_latents"
if ! validate_cache_experiment_against_ae "${AE_RUN_DIR}" "${EXPERIMENT}" "${REPO_ROOT}"; then
    echo "Cache-latents experiment config mismatch vs AE training config" >&2
    exit 1
fi

if [[ -d "$cache_workdir" ]]; then
    echo "Warning: cache workdir already exists, will overwrite: $cache_workdir" >&2
fi

echo "Submitting f8 latent cache job"
echo "  datamodule: ${DATAMODULE}"
echo "  local_experiment: ${EXPERIMENT}"
echo "  autoencoder run: ${AE_RUN_DIR}"
echo "  cache workdir: ${cache_workdir}"

uv run autocast cache-latents --mode slurm \
    --workdir "${cache_workdir}" \
    --output-dir "${cache_workdir}" \
    autoencoder_checkpoint="${ckpt}" \
    local_experiment="${EXPERIMENT}" \
    hydra.launcher.timeout_min=120
