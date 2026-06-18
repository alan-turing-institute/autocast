#!/bin/bash

set -euo pipefail
# Final 24h autoencoder runs for 4 target datasets.
# Uses per-dataset local_experiment configs (periodic BCs where appropriate)
# with the pixel_shuffle encoder/decoder variant (matches LOLA App. B).
#
# Fixed cosine schedule across datasets (LOLA App. B methodology):
# all datasets train for the same number of epochs, trading unused compute
# on faster datasets for schedule parity (identical LR warmup + decay).
#
# COSINE_EPOCHS=512 is set by gray_scott (the slowest per-epoch dataset);
# CNS and GPE use ~58% of their 24h budget, advection_diffusion ~86%.
#
# Raw per-epoch times (seconds) from TrainingTimerCallback in timing.ckpt
# (2026-04-16 timing runs, batch_size=16, 4-GPU DDP, pixel_shuffle=false):
#   advection_diffusion:       361, 180, 141, 129, 130  (mean 2-4: 133)
#   conditioned_navier_stokes: 340, 145, 104,  88,  82  (mean 2-4:  91)
#   gpe_laser_only_wake:       377, 141,  98,  87,  81  (mean 2-4:  89)
#   gray_scott:                407, 201, 160, 147, 154  (mean 2-4: 154)
#
# Gray_scott fit check: 512 * 154 + 607 (warmup) = 79455s = 92% of 24h,
# leaving ~8% buffer for pixel_shuffle overhead and timing variance.
# PSGD's preconditioner update prob decays 1.0 → 0.03 over ~4000 steps,
# so true long-run steady-state is expected 5-15% faster than the 5-epoch
# timing suggests — making 512 safer than the naive calc.
#
# Even if gray_scott truncates near epoch ~495, cos(0.967π) ≈ -0.994
# means LR is already at ~0.3% of max: practically at LR_min.
COSINE_EPOCHS=512
BUDGET_MAX_TIME="01:00:00:00"
# SLURM timeout with 30-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

# Per-dataset local_experiment configs
# (each enables pixel_shuffle via a 2-line model.{encoder,decoder} override).
declare -A EXPERIMENTS=(
    ["gray_scott"]="ae/gray_scott/ae_dc_large"
    ["gpe_laser_only_wake"]="ae/gpe_laser_wake_only/ae_dc_large"
    ["conditioned_navier_stokes"]="ae/conditioned_navier_stokes/ae_dc_large"
    ["advection_diffusion"]="ae/advection_diffusion/ae_dc_large"
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

        echo "Submitting autoencoder training"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${COSINE_EPOCHS}"

        uv run autocast ae --mode slurm "${dry_run_arg[@]}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            logging.wandb.enabled=true \
            +optimizer.cosine_epochs="${COSINE_EPOCHS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${COSINE_EPOCHS}"
    done
done