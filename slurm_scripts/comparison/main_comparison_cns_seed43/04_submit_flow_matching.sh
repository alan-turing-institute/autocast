#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_REPO_ROOT}/slurm_scripts/comparison/cached_latents/validate_cached_latents_against_ae.sh"

SUBMIT="${SUBMIT:-false}"
COSINE_EPOCHS=3223
TIMEOUT_MIN=1439
QUARTER_EPOCHS=$((COSINE_EPOCHS / 4))

require_boolean SUBMIT "${SUBMIT}"
require_dataset_complete
for split in train valid test; do
    if [[ ! -d "${CACHE_DIR}/${split}" ]]; then
        echo "Step 3 dependency is incomplete; missing: ${CACHE_DIR}/${split}" >&2
        exit 1
    fi
done
validate_cached_latents_against_ae "${AE_RUN_DIR}"
validate_cached_latents_complete
refuse_existing_path "${FM_RUN_DIR}"
source_commit="$(pinned_source_commit)"

print_pipeline_paths
echo "Step 4 plan"
echo "  mode: sbatch via the AutoCast Slurm launcher"
echo "  workdir: ${FM_RUN_DIR}"
echo "  config: ${FM_EXPERIMENT}"
echo "  cached data: ${CACHE_DIR}"
echo "  GPUs: 4"
echo "  training seed: ${TRAINING_SEED}"
echo "  AutoCast source: ${source_commit}"
echo "  epochs/cosine epochs: ${COSINE_EPOCHS}/${COSINE_EPOCHS}"
echo "  trainer budget: 00:23:59:00"

dry_run=(--dry-run)
source_dir="${PIPELINE_REPO_ROOT}"
if [[ "${SUBMIT}" == "true" ]]; then
    dry_run=()
    source_dir="$(prepare_autocast_source)"
fi

cd "${source_dir}"
uv run --frozen autocast processor --mode slurm "${dry_run[@]}" \
    --workdir "${FM_RUN_DIR}" \
    local_experiment="${FM_EXPERIMENT}" \
    datamodule.data_path="${CACHE_DIR}" \
    seed="${TRAINING_SEED}" \
    logging.wandb.enabled=true \
    +provenance.source_commit="${source_commit}" \
    +provenance.reference_source_commit="${REFERENCE_FM_SOURCE_COMMIT}" \
    +provenance.reference_run=diff_cns64_flow_matching_vit_09490da_636fcc3 \
    +provenance.dataset_seed="${DATASET_SEED}" \
    optimizer.cosine_epochs="${COSINE_EPOCHS}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
    trainer.max_time=00:23:59:00 \
    +trainer.max_epochs="${COSINE_EPOCHS}" \
    trainer.callbacks.0.every_n_epochs="${QUARTER_EPOCHS}" \
    trainer.callbacks.0.save_top_k=-1 \
    'trainer.callbacks.0.filename="quarter-{epoch:04d}"'

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preview only. Run with SUBMIT=true after the cache has been reviewed."
fi
