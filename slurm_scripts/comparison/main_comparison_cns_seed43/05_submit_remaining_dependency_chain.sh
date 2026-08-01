#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"

SUBMIT="${SUBMIT:-false}"
AE_JOB_ID="${AE_JOB_ID:-5861060}"
readonly CACHE_BATCH_SCRIPT="${PIPELINE_SCRIPT_DIR}/03_cache_latents_after_ae_batch.sh"
readonly FM_BATCH_SCRIPT="${PIPELINE_SCRIPT_DIR}/04_flow_matching_after_cache_batch.sh"

require_boolean SUBMIT "${SUBMIT}"
if [[ ! "${AE_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "AE_JOB_ID must be numeric, got: ${AE_JOB_ID}" >&2
    exit 2
fi
require_dataset_complete
refuse_existing_path "${CACHE_DIR}"
refuse_existing_path "${FM_RUN_DIR}"
for script in "${CACHE_BATCH_SCRIPT}" "${FM_BATCH_SCRIPT}"; do
    if [[ ! -x "${script}" ]]; then
        echo "Missing executable batch script: ${script}" >&2
        exit 1
    fi
done

if [[ "${SUBMIT}" == "true" ]]; then
    require_current_source_ready
fi
source_commit="$(current_source_commit)"

echo "Remaining new-AE dependency chain"
echo "  AE job: ${AE_JOB_ID}"
echo "  cache dependency: afterany:${AE_JOB_ID}"
echo "  cache output: ${CACHE_DIR}"
echo "  FM dependency: afterok:<cache-job-id>"
echo "  FM output: ${FM_RUN_DIR}"
echo "  AutoCast project: ${PIPELINE_REPO_ROOT}"
echo "  AutoCast source: ${source_commit}"

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preview only. Run with SUBMIT=true after review and commit."
    exit
fi

mkdir -p "${LOG_DIR}"
cache_job_id="$({
    sbatch --parsable \
        --dependency="afterany:${AE_JOB_ID}" \
        --export="ALL,PIPELINE_EXPECTED_SOURCE_COMMIT=${source_commit}" \
        --chdir="${PIPELINE_REPO_ROOT}" \
        --output="${LOG_DIR}/new-ae-cache-%j.out" \
        --error="${LOG_DIR}/new-ae-cache-%j.err" \
        "${CACHE_BATCH_SCRIPT}"
} | cut -d ';' -f 1)"
if [[ ! "${cache_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse submitted cache job ID: ${cache_job_id}" >&2
    exit 1
fi

if ! fm_job_id="$({
    sbatch --parsable \
        --dependency="afterok:${cache_job_id}" \
        --export="ALL,PIPELINE_EXPECTED_SOURCE_COMMIT=${source_commit}" \
        --chdir="${PIPELINE_REPO_ROOT}" \
        --output="${LOG_DIR}/new-ae-fm-%j.out" \
        --error="${LOG_DIR}/new-ae-fm-%j.err" \
        "${FM_BATCH_SCRIPT}"
} | cut -d ';' -f 1)"; then
    echo "Cache job ${cache_job_id} was submitted, but FM submission failed." >&2
    exit 1
fi
if [[ ! "${fm_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Cache job ${cache_job_id} was submitted." >&2
    echo "Could not parse submitted FM job ID: ${fm_job_id}" >&2
    exit 1
fi

echo "Submitted cache job ${cache_job_id} afterany:${AE_JOB_ID}"
echo "Submitted FM job ${fm_job_id} afterok:${cache_job_id}"
