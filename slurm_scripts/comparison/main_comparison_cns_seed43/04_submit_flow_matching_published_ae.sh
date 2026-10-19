#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"
source "${PIPELINE_SCRIPT_DIR}/validate_published_ae_cache.sh"

SUBMIT="${SUBMIT:-false}"
COSINE_EPOCHS=3223
TIMEOUT_MIN=1439
QUARTER_EPOCHS=$((COSINE_EPOCHS / 4))

require_boolean SUBMIT "${SUBMIT}"
require_dataset_complete
for required in \
    "${PUBLISHED_AE_CACHE_DIR}/train" \
    "${PUBLISHED_AE_CACHE_DIR}/valid" \
    "${PUBLISHED_AE_CACHE_DIR}/test" \
    "${PUBLISHED_AE_CACHE_DIR}/autoencoder_config.yaml"; do
    if [[ ! -e "${required}" ]]; then
        echo "Published-AE cache dependency is incomplete; missing: ${required}" >&2
        exit 1
    fi
done
validate_published_ae_cache_output \
    "${PUBLISHED_AE_CACHE_DIR}" "${PUBLISHED_AE_RUN_DIR}" "${DATASET_DIR}"
validate_cached_latents_complete \
    "${PIPELINE_REPO_ROOT}" "${PUBLISHED_AE_CACHE_DIR}"
refuse_existing_path "${PUBLISHED_AE_FM_RUN_DIR}"

if [[ "${SUBMIT}" == "true" ]]; then
    require_current_source_ready
fi
source_commit="$(current_source_commit)"

echo "Published-AE FM plan"
echo "  mode: sbatch via the AutoCast Slurm launcher"
echo "  workdir: ${PUBLISHED_AE_FM_RUN_DIR}"
echo "  config: ${PUBLISHED_AE_FM_EXPERIMENT}"
echo "  cached data: ${PUBLISHED_AE_CACHE_DIR}"
echo "  raw-data identity: ${DATASET_DIR}"
echo "  cache source AE: ${PUBLISHED_AE_RUN_DIR}"
echo "  GPUs/tasks/nodes: 4/4/1"
echo "  training seed: ${TRAINING_SEED}"
echo "  AutoCast source: ${source_commit}"
echo "  epochs/cosine epochs: ${COSINE_EPOCHS}/${COSINE_EPOCHS}"
echo "  trainer budget: 00:23:59:00"

dry_run=(--dry-run)
if [[ "${SUBMIT}" == "true" ]]; then
    dry_run=()
fi

cd "${PIPELINE_REPO_ROOT}"
uv run --project "${PIPELINE_REPO_ROOT}" --frozen --no-sync \
    autocast processor --mode slurm "${dry_run[@]}" \
    --workdir "${PUBLISHED_AE_FM_RUN_DIR}" \
    local_experiment="${PUBLISHED_AE_FM_EXPERIMENT}" \
    datamodule.data_path="${PUBLISHED_AE_CACHE_DIR}" \
    seed="${TRAINING_SEED}" \
    logging.wandb.enabled=true \
    +provenance.source_commit="${source_commit}" \
    +provenance.reference_source_commit="${REFERENCE_FM_SOURCE_COMMIT}" \
    +provenance.reference_run=diff_cns64_flow_matching_vit_09490da_636fcc3 \
    +provenance.cache_source_run=ae_cns64_3a7999b_b9c29f8 \
    +provenance.dataset_seed="${DATASET_SEED}" \
    optimizer.cosine_epochs="${COSINE_EPOCHS}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
    +hydra.launcher.additional_parameters.nodes=1 \
    trainer.max_time=00:23:59:00 \
    +trainer.max_epochs="${COSINE_EPOCHS}" \
    trainer.callbacks.0.every_n_epochs="${QUARTER_EPOCHS}" \
    trainer.callbacks.0.save_top_k=-1 \
    'trainer.callbacks.0.filename="quarter-{epoch:04d}"'

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preview only. Run with SUBMIT=true after the cache has been reviewed."
fi
