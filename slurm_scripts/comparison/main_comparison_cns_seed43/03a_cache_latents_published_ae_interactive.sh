#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_SCRIPT_DIR}/validate_published_ae_cache.sh"

RUN="${RUN:-false}"
RESERVATION="${RESERVATION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

require_boolean RUN "${RUN}"
require_dataset_complete
for required in \
    "${PUBLISHED_AE_CHECKPOINT}" \
    "${PUBLISHED_AE_RUN_DIR}/resolved_autoencoder_config.yaml" \
    "${PUBLISHED_DATASET_DIR}/stats.yml"; do
    if [[ ! -f "${required}" ]]; then
        echo "Published AE dependency is incomplete; missing: ${required}" >&2
        exit 1
    fi
done
refuse_existing_path "${PUBLISHED_AE_CACHE_DIR}"
validate_published_ae_cache_experiment \
    "${PUBLISHED_AE_RUN_DIR}" \
    "${PUBLISHED_AE_CACHE_EXPERIMENT}" \
    "${PIPELINE_REPO_ROOT}" \
    "${DATASET_DIR}"

if [[ "${RUN}" == "true" ]]; then
    require_current_source_ready
fi
source_commit="$(current_source_commit)"

echo "Published-AE cache plan"
echo "  mode: srun, reservation=${RESERVATION}"
echo "  raw data: ${DATASET_DIR}"
echo "  checkpoint: ${PUBLISHED_AE_CHECKPOINT}"
echo "  normalization: ${PUBLISHED_DATASET_DIR}/stats.yml"
echo "  workdir/output: ${PUBLISHED_AE_CACHE_DIR}"
echo "  config: ${PUBLISHED_AE_CACHE_EXPERIMENT}"
echo "  GPUs/tasks/nodes: 1/1/1"
echo "  AutoCast source: ${source_commit}"

if [[ "${RUN}" == "false" ]]; then
    echo "Preview only. Run with RUN=true after review and commit."
    exit
fi

mkdir -p "${LOG_DIR}"
srun \
    --reservation="${RESERVATION}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --kill-on-bad-exit=1 \
    --job-name="cns_seed43_cache_published_ae" \
    --output="${LOG_DIR}/published-ae-cache-%j.out" \
    --error="${LOG_DIR}/published-ae-cache-%j.err" \
    --chdir="${PIPELINE_REPO_ROOT}" \
    uv run --project "${PIPELINE_REPO_ROOT}" --frozen --no-sync \
        autocast cache-latents --mode local \
        --workdir "${PUBLISHED_AE_CACHE_DIR}" \
        --output-dir "${PUBLISHED_AE_CACHE_DIR}" \
        autoencoder_checkpoint="${PUBLISHED_AE_CHECKPOINT}" \
        local_experiment="${PUBLISHED_AE_CACHE_EXPERIMENT}" \
        +provenance.source_commit="${source_commit}" \
        +provenance.reference_ae_source_commit="${REFERENCE_AE_SOURCE_COMMIT}" \
        +provenance.cache_source_run=ae_cns64_3a7999b_b9c29f8 \
        +provenance.dataset_seed="${DATASET_SEED}" \
        +provenance.normalization_source_dataset=conditioned_navier_stokes_2d_5e1f575

validate_published_ae_cache_output \
    "${PUBLISHED_AE_CACHE_DIR}" "${PUBLISHED_AE_RUN_DIR}" "${DATASET_DIR}"
validate_cached_latents_complete \
    "${PIPELINE_REPO_ROOT}" "${PUBLISHED_AE_CACHE_DIR}"
