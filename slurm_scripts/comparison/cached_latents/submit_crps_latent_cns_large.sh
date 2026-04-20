#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/validate_cached_latents_against_ae.sh"
# Ablation: final 24h CRPS-in-cached-latent run for
# conditioned_navier_stokes only.
# Model: AzulaViTProcessor / vit_azula_large (hidden_dim=568, n_layers=12,
# num_heads=8, patch_size=1, n_noise_channels=1024). Head: AlphaFairCRPSLoss,
# n_members=8. Optimizer: adamw_half (LR=2e-4, warmup=0). Batch: 32/GPU x
# n_members=8 internal expansion.
#
# Replace COSINE_EPOCHS once timing is available from:
#   submit_crps_latent_cns_timing.sh

DATAMODULE="conditioned_navier_stokes"
EXPERIMENT="processor/conditioned_navier_stokes/crps_vit_azula_large"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
COSINE_EPOCHS=345  # 245.0 s/ep (timing_efficient_crps, 2026-04-18)

BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

cache_dir="${AE_RUN_DIR}/cached_latents"

if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
    echo "Skipping ${DATAMODULE}: cache missing train/valid/test under ${cache_dir}" >&2
    exit 1
fi
if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
    echo "Skipping ${DATAMODULE}: cached-latents config mismatch vs AE training config" >&2
    exit 1
fi

# Save checkpoints at 25/50/75/100% of the schedule (top_k=-1 keeps all).
# save_last: true (set in trainer/default.yaml) ensures last.ckpt captures
# the final epoch even if it doesn't land on a quarter boundary.
quarter_epochs=$((COSINE_EPOCHS / 4))

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting CRPS-in-latent ablation training"
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
        trainer.callbacks.0.every_n_epochs="${quarter_epochs}" \
        trainer.callbacks.0.save_top_k=-1 \
        trainer.callbacks.0.filename=\"quarter-{epoch:04d}\"
done
