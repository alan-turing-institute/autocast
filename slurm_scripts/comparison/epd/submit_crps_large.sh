#!/bin/bash

set -euo pipefail
# Final 24h CRPS-in-ambient runs for 4 target datasets.
# Model: vit_azula_large (hidden_dim=568, n_layers=12, num_heads=8,
# patch_size=4, n_noise_channels=1024). Head: AlphaFairCRPSLoss, n_members=8.
# See local_hydra/local_experiment/epd/<dataset>/crps_vit_azula_large.yaml
# for the authoritative hyperparameters.
#
# Per-dataset cosine schedule: each (method, dataset) pair fills its own
# 24h budget so each model gets its best shot within budget. Values from
# submit_crps_timing.sh (2026-04-18) via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
#
# learning_rate (2e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
declare -A COSINE_EPOCHS_BY_DATASET=(
    ["gray_scott"]=399                  # 212.0 s/ep
    ["gpe_laser_only_wake"]=477         # 177.2 s/ep
    ["conditioned_navier_stokes"]=473   # 178.8 s/ep
    ["advection_diffusion"]=478         # 177.0 s/ep
)
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
    cosine_epochs="${COSINE_EPOCHS_BY_DATASET[$datamodule]}"
    # Save checkpoints every ~5% of optimizer-step progress (top_k=-1 keeps all).
    # save_last: true (set in trainer/default.yaml) ensures last.ckpt captures
    # the final state even if it doesn't land exactly on a progress boundary.

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
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            logging.wandb.enabled=true \
            optimizer.cosine_epochs="${cosine_epochs}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${cosine_epochs}" \
            trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
            +trainer.callbacks.0.every_n_epochs=0 \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
    done
done
