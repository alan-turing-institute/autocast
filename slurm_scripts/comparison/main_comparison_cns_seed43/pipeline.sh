#!/bin/bash

# Shared, immutable identity for the CNS seed-43 main-comparison repeat.

PIPELINE_SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
PIPELINE_REPO_ROOT="$(
    cd "${PIPELINE_SCRIPT_DIR}/../../.."
    pwd
)"

readonly PIPELINE_SCRIPT_DIR
readonly PIPELINE_REPO_ROOT
readonly PIPELINE_ID="main_comparison_cns_seed43"
readonly DATASET_SEED="43"
readonly TRAINING_SEED="42"
readonly DATASET_DIR="/projects/u6eo/autocast/datasets/conditioned_navier_stokes_2d_seed43_20260801"
readonly PUBLISHED_DATASET_DIR="/projects/u6eo/autocast/datasets/conditioned_navier_stokes_2d_5e1f575"
readonly RUN_ROOT="${PIPELINE_REPO_ROOT}/outputs/2026-08-01/${PIPELINE_ID}"
readonly CRPS_RUN_DIR="${RUN_ROOT}/crps_vit_azula_large"
readonly AE_RUN_DIR="${RUN_ROOT}/ae_dc_large"
readonly CACHE_DIR="${AE_RUN_DIR}/cached_latents"
readonly FM_RUN_DIR="${RUN_ROOT}/fm_vit_large"
readonly PUBLISHED_AE_RUN_DIR="${PIPELINE_REPO_ROOT}/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
readonly PUBLISHED_AE_CHECKPOINT="${PUBLISHED_AE_RUN_DIR}/autoencoder.ckpt"
readonly PUBLISHED_AE_CACHE_DIR="${RUN_ROOT}/published_ae/cached_latents"
readonly PUBLISHED_AE_FM_RUN_DIR="${RUN_ROOT}/fm_vit_large_published_ae"
readonly LOG_DIR="${RUN_ROOT}/slurm_logs"
readonly SOURCE_COMMIT_FILE="${DATASET_DIR}/autocast_source_commit.txt"
readonly DATASET_VALIDATION_MARKER="${DATASET_DIR}/validation_complete.txt"

# Commits encoded in the three published run directory names. These identify
# the reference experiments; rerun jobs use the committed current checkout.
readonly REFERENCE_CRPS_SOURCE_COMMIT="bed4611609d224bb3497e858ba278d028e7430d2"
readonly REFERENCE_AE_SOURCE_COMMIT="3a7999b733254d6a9e572644be3c694744c07305"
readonly REFERENCE_FM_SOURCE_COMMIT="09490dad1093b304a69c0b2d14695887c536e67f"
readonly REQUIRED_DDP_FIX_COMMIT="f5ee48356934e0e80f9da77c0922edcb8a4cf4b7"

readonly AUTOSIM_REPO="/home/u6eo/ltcx7228.u6eo/autosim"
# This is the last AutoSim commit before generation of the published dataset
# started at 2026-03-07 18:07 UTC. Its resolved CNS preset is byte-identical.
readonly AUTOSIM_GENERATION_COMMIT="bd5ac317c3533a18a52fb07e05abe3a0247b437d"
# This commit introduced the exact four-significant-figure stats.yml procedure
# used on the published dataset on 2026-03-11.
readonly AUTOSIM_STATS_COMMIT="0bc366adf92fac25228da7550f70d58819f37708"

readonly CRPS_EXPERIMENT="reruns/main_comparison_cns_seed43/crps_vit_azula_large"
readonly AE_EXPERIMENT="reruns/main_comparison_cns_seed43/ae_dc_large"
readonly CACHE_EXPERIMENT="reruns/main_comparison_cns_seed43/cache_latents"
readonly FM_EXPERIMENT="reruns/main_comparison_cns_seed43/fm_vit_large"
readonly PUBLISHED_AE_CACHE_EXPERIMENT="reruns/main_comparison_cns_seed43/cache_latents_published_ae"
readonly PUBLISHED_AE_FM_EXPERIMENT="reruns/main_comparison_cns_seed43/fm_vit_large_published_ae"

require_boolean() {
    local name="$1"
    local value="$2"
    if [[ "${value}" != "true" && "${value}" != "false" ]]; then
        echo "${name} must be true or false, got: ${value}" >&2
        return 2
    fi
}

refuse_existing_path() {
    local path="$1"
    if [[ -e "${path}" || -L "${path}" ]]; then
        echo "Refusing to reuse or overwrite existing path: ${path}" >&2
        return 1
    fi
}

require_dataset_complete() {
    local required
    for required in \
        "${DATASET_DIR}/train/data.pt" \
        "${DATASET_DIR}/valid/data.pt" \
        "${DATASET_DIR}/test/data.pt" \
        "${DATASET_DIR}/stats.yml" \
        "${DATASET_DIR}/resolved_config.yaml" \
        "${DATASET_DIR}/provenance.yml" \
        "${SOURCE_COMMIT_FILE}" \
        "${DATASET_VALIDATION_MARKER}"; do
        if [[ ! -f "${required}" ]]; then
            echo "Dataset dependency is incomplete; missing: ${required}" >&2
            return 1
        fi
    done
}

require_current_source_ready() {
    if ! git -C "${PIPELINE_REPO_ROOT}" merge-base --is-ancestor \
        "${REQUIRED_DDP_FIX_COMMIT}" HEAD; then
        echo "Current branch does not contain the required DDP teardown fix." >&2
        return 1
    fi
    if [[ -n "$(git -C "${PIPELINE_REPO_ROOT}" status --porcelain)" ]]; then
        echo "Refusing to run from an uncommitted AutoCast working tree." >&2
        return 1
    fi
}

current_source_commit() {
    git -C "${PIPELINE_REPO_ROOT}" rev-parse HEAD
}

validate_cached_latents_complete() {
    local project_dir="${1:-${PIPELINE_REPO_ROOT}}"
    local cache_dir="${2:-${CACHE_DIR}}"
    uv run --project "${project_dir}" --frozen --no-sync python \
        "${PIPELINE_SCRIPT_DIR}/validate_cached_latents.py" "${cache_dir}"
}

print_pipeline_paths() {
    echo "Pipeline: ${PIPELINE_ID}"
    echo "  dataset: ${DATASET_DIR}"
    echo "  CRPS: ${CRPS_RUN_DIR}"
    echo "  autoencoder: ${AE_RUN_DIR}"
    echo "  cached latents: ${CACHE_DIR}"
    echo "  flow matching: ${FM_RUN_DIR}"
    echo "  published-AE cached latents: ${PUBLISHED_AE_CACHE_DIR}"
    echo "  published-AE flow matching: ${PUBLISHED_AE_FM_RUN_DIR}"
}
