#!/bin/bash
#SBATCH --job-name=cns_seed43_cache_new_ae
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --mem=0

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_REPO_ROOT}/slurm_scripts/comparison/cached_latents/validate_cached_latents_against_ae.sh"

readonly AE_CHECKPOINT="${AE_RUN_DIR}/autoencoder.ckpt"

require_expected_source_ready
require_dataset_complete
for required in \
    "${AE_CHECKPOINT}" \
    "${AE_RUN_DIR}/resolved_autoencoder_config.yaml"; do
    if [[ ! -f "${required}" ]]; then
        echo "Completed AE dependency is missing: ${required}" >&2
        exit 1
    fi
done
refuse_existing_path "${CACHE_DIR}"
validate_cache_experiment_against_ae \
    "${AE_RUN_DIR}" "${CACHE_EXPERIMENT}" "${PIPELINE_REPO_ROOT}"

source_commit="$(current_source_commit)"
echo "Short new-AE cache batch job"
echo "  raw data: ${DATASET_DIR}"
echo "  checkpoint: ${AE_CHECKPOINT}"
echo "  output: ${CACHE_DIR}"
echo "  walltime: 00:10:00"
echo "  AutoCast source: ${source_commit}"

cd "${PIPELINE_REPO_ROOT}"
uv run --project "${PIPELINE_REPO_ROOT}" --frozen --no-sync \
    srun python -m autocast.scripts.cache_latents \
        hydra.run.dir="${CACHE_DIR}" \
        +cache_latents.output_dir="${CACHE_DIR}" \
        autoencoder_checkpoint="${AE_CHECKPOINT}" \
        local_experiment="${CACHE_EXPERIMENT}" \
        +provenance.source_commit="${source_commit}" \
        +provenance.reference_ae_source_commit="${REFERENCE_AE_SOURCE_COMMIT}" \
        +provenance.dataset_seed="${DATASET_SEED}"

validate_cached_latents_against_ae "${AE_RUN_DIR}"
validate_cached_latents_complete "${PIPELINE_REPO_ROOT}" "${CACHE_DIR}"
