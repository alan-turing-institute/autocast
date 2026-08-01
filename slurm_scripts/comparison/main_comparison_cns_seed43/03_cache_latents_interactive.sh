#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_REPO_ROOT}/slurm_scripts/comparison/cached_latents/validate_cached_latents_against_ae.sh"

RUN="${RUN:-false}"
RESERVATION="${RESERVATION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
AE_CHECKPOINT="${AE_RUN_DIR}/autoencoder.ckpt"

require_boolean RUN "${RUN}"
require_dataset_complete
if [[ ! -f "${AE_CHECKPOINT}" ]]; then
    echo "Step 2 dependency is incomplete; missing: ${AE_CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${AE_RUN_DIR}/resolved_autoencoder_config.yaml" ]]; then
    echo "Step 2 dependency is incomplete; missing resolved AE config" >&2
    exit 1
fi
refuse_existing_path "${CACHE_DIR}"
validate_cache_experiment_against_ae \
    "${AE_RUN_DIR}" "${CACHE_EXPERIMENT}" "${PIPELINE_REPO_ROOT}"
source_commit="$(pinned_source_commit)"

print_pipeline_paths
echo "Step 3 plan"
echo "  mode: srun, reservation=${RESERVATION}"
echo "  workdir/output: ${CACHE_DIR}"
echo "  config: ${CACHE_EXPERIMENT}"
echo "  checkpoint: ${AE_CHECKPOINT}"
echo "  GPUs: 1"
echo "  AutoCast source: ${source_commit}"

if [[ "${RUN}" == "false" ]]; then
    echo "Preview only. Run with RUN=true after the AE has completed and been reviewed."
    exit
fi

source_dir="$(prepare_autocast_source)"
mkdir -p "${LOG_DIR}"
srun \
    --reservation="${RESERVATION}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --kill-on-bad-exit=1 \
    --job-name="cns_seed43_cache" \
    --output="${LOG_DIR}/cache-%j.out" \
    --error="${LOG_DIR}/cache-%j.err" \
    --chdir="${source_dir}" \
    uv run --frozen autocast cache-latents --mode local \
        --workdir "${CACHE_DIR}" \
        --output-dir "${CACHE_DIR}" \
        autoencoder_checkpoint="${AE_CHECKPOINT}" \
        local_experiment="${CACHE_EXPERIMENT}" \
        +provenance.source_commit="${source_commit}" \
        +provenance.reference_ae_source_commit="${REFERENCE_AE_SOURCE_COMMIT}" \
        +provenance.dataset_seed="${DATASET_SEED}"

validate_cached_latents_against_ae "${AE_RUN_DIR}"
validate_cached_latents_complete "${source_dir}"
