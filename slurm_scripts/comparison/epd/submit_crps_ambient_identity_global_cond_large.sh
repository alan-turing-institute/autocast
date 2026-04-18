#!/bin/bash

set -euo pipefail
# Final 24h CRPS-in-ambient ablation for conditioned_navier_stokes.
# Model: vit_azula_large (hidden_dim=568, n_layers=12, num_heads=8,
# patch_size=4, n_noise_channels=1024), n_members=8, AlphaFairCRPSLoss.
# Ablation path: identity encoder/decoder + processor include_global_cond=true
# (conditioning via global_cond/AdaLN, not spatial concatenation).
# See local_hydra/local_experiment/epd/conditioned_navier_stokes/
# crps_vit_azula_large_identity_global_cond.yaml for authoritative settings.
#
# Replace COSINE_EPOCHS value after running:
#   submit_crps_ambient_identity_global_cond_timing.sh
# and then extracting:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
declare -A COSINE_EPOCHS_BY_DATASET=(
    ["conditioned_navier_stokes"]=469  # 180.3 s/ep (timing_efficient_crps, 2026-04-18)
)
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large_identity_global_cond"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    cosine_epochs="${COSINE_EPOCHS_BY_DATASET[$datamodule]}"
    # Save checkpoints at 25/50/75/100% of the schedule (top_k=-1 keeps all).
    # save_last: true (set in trainer/default.yaml) ensures last.ckpt captures
    # the final epoch even if it doesn't land on a quarter boundary.
    quarter_epochs=$((cosine_epochs / 4))

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS ambient identity+global_cond training"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            local_experiment="${experiment}" \
            logging.wandb.enabled=true \
            optimizer.cosine_epochs="${cosine_epochs}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${cosine_epochs}" \
            trainer.callbacks.0.every_n_epochs="${quarter_epochs}" \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename="quarter-{epoch:04d}"
    done
done