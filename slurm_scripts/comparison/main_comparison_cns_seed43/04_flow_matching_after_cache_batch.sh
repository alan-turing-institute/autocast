#!/bin/bash
#SBATCH --job-name=cns_seed43_fm_new_ae
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=23:59:00
#SBATCH --mem=0

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_REPO_ROOT}/slurm_scripts/comparison/cached_latents/validate_cached_latents_against_ae.sh"

readonly COSINE_EPOCHS=3223
readonly QUARTER_EPOCHS=$((COSINE_EPOCHS / 4))

require_expected_source_ready
require_dataset_complete
for split in train valid test; do
    if [[ ! -d "${CACHE_DIR}/${split}" ]]; then
        echo "Completed cache dependency is missing: ${CACHE_DIR}/${split}" >&2
        exit 1
    fi
done
validate_cached_latents_against_ae "${AE_RUN_DIR}"
validate_cached_latents_complete "${PIPELINE_REPO_ROOT}" "${CACHE_DIR}"
refuse_existing_path "${FM_RUN_DIR}"

source_commit="$(current_source_commit)"
echo "New-AE FM batch job"
echo "  cache: ${CACHE_DIR}"
echo "  output: ${FM_RUN_DIR}"
echo "  GPUs/tasks/nodes: 4/4/1"
echo "  training seed: ${TRAINING_SEED}"
echo "  AutoCast source: ${source_commit}"
echo "  epochs/cosine epochs: ${COSINE_EPOCHS}/${COSINE_EPOCHS}"

cd "${PIPELINE_REPO_ROOT}"
exec uv run --project "${PIPELINE_REPO_ROOT}" --frozen --no-sync \
    srun python -m autocast.scripts.train.processor \
        logging.wandb.name=fm_vit_large \
        local_experiment="${FM_EXPERIMENT}" \
        datamodule.data_path="${CACHE_DIR}" \
        seed="${TRAINING_SEED}" \
        logging.wandb.enabled=true \
        +provenance.source_commit="${source_commit}" \
        +provenance.reference_source_commit="${REFERENCE_FM_SOURCE_COMMIT}" \
        +provenance.reference_run=diff_cns64_flow_matching_vit_09490da_636fcc3 \
        +provenance.dataset_seed="${DATASET_SEED}" \
        optimizer.cosine_epochs="${COSINE_EPOCHS}" \
        trainer.max_time=00:23:59:00 \
        +trainer.max_epochs="${COSINE_EPOCHS}" \
        trainer.callbacks.0.every_n_epochs="${QUARTER_EPOCHS}" \
        trainer.callbacks.0.save_top_k=-1 \
        'trainer.callbacks.0.filename="quarter-{epoch:04d}"' \
        hydra.run.dir="${FM_RUN_DIR}"
