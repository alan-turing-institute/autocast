#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../comparison/cached_latents/validate_cached_latents_against_ae.sh"
# CNS-only AE-compression ablation: 24h FM-in-latent run on the f8 (8x8x8)
# cached latents.
# Mirrors slurm_scripts/comparison/cached_latents/submit_fm_large.sh but
# targets the single ae_dc_large_f8 run.
#
# Replace COSINE_EPOCHS once timing is available from:
#   submit_fm_f8_timing.sh

DATAMODULE="conditioned_navier_stokes"
EXPERIMENT="processor/conditioned_navier_stokes/fm_vit_large"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-26/ae_cns64_de1b4b7_e1059d7"
COSINE_EPOCHS=3223  # placeholder: 16x16 main-study value; re-fit from f8 timing

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

cache_dir="${AE_RUN_DIR}/cached_latents"

if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
    echo "Cache missing train/valid/test under ${cache_dir}" >&2
    exit 1
fi
if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
    echo "Cached-latents config mismatch vs AE training config" >&2
    exit 1
fi

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting f8 FM-in-latent training"
    echo "  mode: ${run_label}"
    echo "  datamodule: ${DATAMODULE}"
    echo "  local_experiment: ${EXPERIMENT}"
    echo "  cache dir: ${cache_dir}"
    echo "  cosine_epochs: ${COSINE_EPOCHS}"

    uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
        local_experiment="${EXPERIMENT}" \
        datamodule.data_path="${cache_dir}" \
        logging.wandb.enabled=true \
        optimizer.cosine_epochs="${COSINE_EPOCHS}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        +trainer.max_epochs="${COSINE_EPOCHS}" \
        trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
done
