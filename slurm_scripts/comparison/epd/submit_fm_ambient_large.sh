#!/bin/bash

set -euo pipefail
# Final 24h FM-in-ambient runs for 4 target datasets.
# Model: flow_matching_vit (vit backbone, hid_channels=704, hid_blocks=12,
# attention_heads=8, patch_size=4, flow_ode_steps=50). Encoder/decoder:
# identity (conditioning flows via backbone global_cond / AdaLN, not spatial
# concatenation — distinct mechanism from CRPS ambient's permute_concat).
# Optimizer: adamw_half (LR=1e-4, warmup=0). Batch size: 256/GPU
# (effective-batch parity with CRPS bs=32 x n_members=8).
# See local_hydra/local_experiment/epd/<dataset>/fm_vit_large.yaml for the
# authoritative hyperparameters.
#
# Per-dataset cosine schedule: each (method, dataset) pair fills its own
# 24h budget so each model gets its best shot within budget. Values from
# submit_fm_ambient_timing.sh (2026-04-18) via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
#
# learning_rate (1e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
declare -A COSINE_EPOCHS_BY_DATASET=(
    ["gray_scott"]=2631                 # 32.2 s/ep
    ["gpe_laser_only_wake"]=3171        # 26.7 s/ep
    ["conditioned_navier_stokes"]=2982  # 28.4 s/ep
    ["advection_diffusion"]=3264        # 25.9 s/ep
)
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

# Per-dataset local_experiment configs.
declare -A EXPERIMENTS=(
    ["gray_scott"]="epd/gray_scott/fm_vit_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/fm_vit_large"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/fm_vit_large"
    ["advection_diffusion"]="epd/advection_diffusion/fm_vit_large"
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

        echo "Submitting FM-in-ambient training"
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
