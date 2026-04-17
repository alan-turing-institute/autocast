#!/bin/bash

set -euo pipefail
# Final 24h CRPS-in-ambient runs for 4 target datasets.
# Model: vit_azula_large (hidden_dim=568, n_layers=12, num_heads=8,
# patch_size=4, n_noise_channels=1024). Head: AlphaFairCRPSLoss, n_members=8.
# See local_hydra/local_experiment/epd/<dataset>/crps_vit_azula_large.yaml
# for the authoritative hyperparameters.
#
# Fixed cosine schedule across datasets (LOLA App. B methodology):
# all datasets train for the same number of epochs.
#
# COSINE_EPOCHS is a placeholder pending timing runs — once
# submit_crps_timing.sh completes and per-epoch times are extracted via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# replace 240 with the recommended value for the slowest dataset.
#
# learning_rate (2e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
COSINE_EPOCHS=240
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

# Per-dataset local_experiment configs.
declare -A EXPERIMENTS=(
    ["gray_scott"]="epd/gray_scott/crps_vit_azula_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
    ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS training"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${COSINE_EPOCHS}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            local_experiment="${experiment}" \
            logging.wandb.enabled=true \
            optimizer.cosine_epochs="${COSINE_EPOCHS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${COSINE_EPOCHS}"
    done
done
