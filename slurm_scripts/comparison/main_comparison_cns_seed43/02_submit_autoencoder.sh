#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"

SUBMIT="${SUBMIT:-false}"
COSINE_EPOCHS=512
TIMEOUT_MIN=1439

require_boolean SUBMIT "${SUBMIT}"
require_dataset_complete
refuse_existing_path "${AE_RUN_DIR}"
if [[ "${SUBMIT}" == "true" ]]; then
    require_current_source_ready
fi
source_commit="$(current_source_commit)"

print_pipeline_paths
echo "Step 2 plan"
echo "  mode: sbatch via the AutoCast Slurm launcher"
echo "  workdir: ${AE_RUN_DIR}"
echo "  config: ${AE_EXPERIMENT}"
echo "  GPUs: 4"
echo "  training seed: ${TRAINING_SEED}"
echo "  AutoCast source: ${source_commit}"
echo "  epochs/cosine epochs: ${COSINE_EPOCHS}/${COSINE_EPOCHS}"
echo "  trainer budget: 01:00:00:00"

dry_run=(--dry-run)
if [[ "${SUBMIT}" == "true" ]]; then
    dry_run=()
fi

cd "${PIPELINE_REPO_ROOT}"
uv run --project "${PIPELINE_REPO_ROOT}" --frozen --no-sync \
    autocast ae --mode slurm "${dry_run[@]}" \
    --workdir "${AE_RUN_DIR}" \
    local_experiment="${AE_EXPERIMENT}" \
    seed="${TRAINING_SEED}" \
    logging.wandb.enabled=true \
    +provenance.source_commit="${source_commit}" \
    +provenance.reference_source_commit="${REFERENCE_AE_SOURCE_COMMIT}" \
    +provenance.reference_run=ae_cns64_3a7999b_b9c29f8 \
    +provenance.dataset_seed="${DATASET_SEED}" \
    +optimizer.cosine_epochs="${COSINE_EPOCHS}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
    trainer.max_time=01:00:00:00 \
    +trainer.max_epochs="${COSINE_EPOCHS}"

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preview only. Run with SUBMIT=true after review."
fi
