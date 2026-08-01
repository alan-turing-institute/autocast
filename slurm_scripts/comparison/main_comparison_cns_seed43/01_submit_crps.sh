#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/pipeline.sh"

SUBMIT="${SUBMIT:-false}"
COSINE_EPOCHS=473
TIMEOUT_MIN=1439

require_boolean SUBMIT "${SUBMIT}"
require_dataset_complete
refuse_existing_path "${CRPS_RUN_DIR}"

source_commit="$(pinned_source_commit)"

print_pipeline_paths
echo "Step 1 plan"
echo "  mode: sbatch via the AutoCast Slurm launcher"
echo "  workdir: ${CRPS_RUN_DIR}"
echo "  config: ${CRPS_EXPERIMENT}"
echo "  GPUs/tasks: 4/4"
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
uv run --frozen autocast epd --mode slurm "${dry_run[@]}" \
    --workdir "${CRPS_RUN_DIR}" \
    local_experiment="${CRPS_EXPERIMENT}" \
    seed="${TRAINING_SEED}" \
    logging.wandb.enabled=true \
    +provenance.source_commit="${source_commit}" \
    +provenance.reference_source_commit="${REFERENCE_CRPS_SOURCE_COMMIT}" \
    +provenance.reference_run=crps_cns64_vit_azula_large_bed4611_c99f534 \
    +provenance.dataset_seed="${DATASET_SEED}" \
    optimizer.cosine_epochs="${COSINE_EPOCHS}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
    trainer.max_time=00:23:59:00 \
    +trainer.max_epochs="${COSINE_EPOCHS}" \
    trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
    trainer.callbacks.0.every_n_epochs=0 \
    trainer.callbacks.0.save_top_k=-1 \
    'trainer.callbacks.0.filename="snapshot-{progress_token}-{epoch:04d}-{step:08d}"'

if [[ "${SUBMIT}" == "false" ]]; then
    echo "Preview only. Run with SUBMIT=true after review."
fi
