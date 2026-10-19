#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"

RUN="${RUN:-false}"
RESERVATION="${RESERVATION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

require_boolean RUN "${RUN}"

run_inside_allocation() {
    refuse_existing_path "${DATASET_DIR}"

    local scratch_dir generation_tree stats_tree source_commit
    source_commit="${PIPELINE_SOURCE_COMMIT:-$(current_source_commit)}"
    if [[ ! "${source_commit}" =~ ^[0-9a-f]{40}$ ]]; then
        echo "Invalid AutoCast source commit: ${source_commit}" >&2
        return 1
    fi
    scratch_dir="$(mktemp -d /tmp/autosim-cns-seed43.XXXXXX)"
    generation_tree="${scratch_dir}/generation"
    stats_tree="${scratch_dir}/stats"
    trap 'rm -rf "${scratch_dir}"' EXIT

    mkdir -p "${generation_tree}" "${stats_tree}"
    git -C "${AUTOSIM_REPO}" archive "${AUTOSIM_GENERATION_COMMIT}" \
        | tar -x -C "${generation_tree}"
    git -C "${AUTOSIM_REPO}" archive "${AUTOSIM_STATS_COMMIT}" \
        | tar -x -C "${stats_tree}"

    echo "Syncing pinned AutoSim generation environment"
    uv sync --project "${generation_tree}" --frozen
    echo "Generating CNS data in: ${DATASET_DIR}"
    uv run --project "${generation_tree}" --frozen autosim \
        simulator=conditioned_navier_stokes_2d \
        dataset.output_dir="${DATASET_DIR}" \
        dataset.n_train=200 \
        dataset.n_valid=20 \
        dataset.n_test=20 \
        seed="${DATASET_SEED}" \
        overwrite=false \
        visualize.enabled=true \
        visualize.split=train \
        'visualize.batch_indices=[0,1,2,3]' \
        visualize.fps=5 \
        visualize.file_ext=mp4 \
        visualize.overwrite=true \
        visualize.preserve_aspect=true

    echo "Syncing pinned AutoSim statistics environment"
    uv sync --project "${stats_tree}" --frozen
    uv run --project "${stats_tree}" --frozen autosim stats \
        "${DATASET_DIR}" \
        --split train \
        --output "${DATASET_DIR}/stats.yml" \
        --field-names smoke,u,v \
        --sig-figs 4

    printf '%s\n' "${source_commit}" > "${SOURCE_COMMIT_FILE}"
    printf '%s\n' \
        "pipeline: ${PIPELINE_ID}" \
        "generated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "dataset_seed: ${DATASET_SEED}" \
        "split_seeds: [${DATASET_SEED}, $((DATASET_SEED + 1)), $((DATASET_SEED + 2))]" \
        "autocast_source_commit: ${source_commit}" \
        "autosim_generation_commit: ${AUTOSIM_GENERATION_COMMIT}" \
        "autosim_stats_commit: ${AUTOSIM_STATS_COMMIT}" \
        "published_reference_dataset: ${PUBLISHED_DATASET_DIR}" \
        > "${DATASET_DIR}/provenance.yml"

    sha256sum \
        "${DATASET_DIR}/train/data.pt" \
        "${DATASET_DIR}/valid/data.pt" \
        "${DATASET_DIR}/test/data.pt" \
        "${DATASET_DIR}/stats.yml" \
        "${DATASET_DIR}/resolved_config.yaml" \
        "${DATASET_DIR}/provenance.yml" \
        "${SOURCE_COMMIT_FILE}" \
        > "${DATASET_DIR}/artifact_sha256.txt"

    uv run --project "${stats_tree}" --frozen python \
        "${PIPELINE_SCRIPT_DIR}/validate_dataset.py" \
        "${DATASET_DIR}" \
        --published-dataset-dir "${PUBLISHED_DATASET_DIR}" \
        --seed "${DATASET_SEED}"

    printf '%s\n' \
        "validated_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "autocast_source_commit: ${source_commit}" \
        > "${DATASET_VALIDATION_MARKER}"
}

print_pipeline_paths
echo "Step 0 plan"
echo "  mode: srun, reservation=${RESERVATION}"
echo "  time: ${TIME_LIMIT}"
echo "  GPUs: 1"
echo "  AutoSim generation commit: ${AUTOSIM_GENERATION_COMMIT}"
echo "  AutoSim stats commit: ${AUTOSIM_STATS_COMMIT}"
echo "  split sizes: 200/20/20"
echo "  tensor geometry: 321 x 64 x 64 x 3"
echo "  dataset seed: ${DATASET_SEED} (split seeds 43/44/45)"

if [[ "${1:-}" == "--inside-allocation" ]]; then
    run_inside_allocation
    exit
fi

refuse_existing_path "${DATASET_DIR}"
if [[ "${RUN}" == "false" ]]; then
    echo "Preview only. Run with RUN=true after reviewing the target above."
    exit
fi

PIPELINE_SOURCE_COMMIT="$(current_source_commit)"
require_current_source_ready
export PIPELINE_SOURCE_COMMIT
mkdir -p "${LOG_DIR}"
srun \
    --reservation="${RESERVATION}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --time="${TIME_LIMIT}" \
    --kill-on-bad-exit=1 \
    --job-name="cns_seed43_data" \
    --output="${LOG_DIR}/data-%j.out" \
    --error="${LOG_DIR}/data-%j.err" \
    bash "${BASH_SOURCE[0]}" --inside-allocation
