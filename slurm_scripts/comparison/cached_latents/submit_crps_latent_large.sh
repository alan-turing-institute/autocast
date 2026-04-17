#!/bin/bash

set -euo pipefail
# Final 24h CRPS-in-latent runs for 4 target datasets.
# Model: AzulaViTProcessor / vit_azula_large (hidden_dim=568, n_layers=12,
# num_heads=8, patch_size=1, n_noise_channels=1024). Head: AlphaFairCRPSLoss,
# n_members=8. Optimizer: adamw_half (LR=2e-4, warmup=0). Batch: 32/GPU x
# n_members=8 internal expansion.
# See local_hydra/local_experiment/processor/<dataset>/crps_vit_azula_large.yaml
# for the authoritative hyperparameters.
#
# COSINE_EPOCHS is a placeholder pending timing runs — once
# submit_crps_latent_timing.sh completes and per-epoch times are extracted via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# replace 1080 with the recommended value for the slowest dataset.
#
# learning_rate (2e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
COSINE_EPOCHS=1080
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

# Per-dataset local_experiment + AE run dir (cached latents live under
# <ae_run_dir>/cached_latents/).
declare -A EXPERIMENTS=(
    ["gray_scott"]="processor/gray_scott/crps_vit_azula_large"
    ["gpe_laser_only_wake"]="processor/gpe_laser_wake_only/crps_vit_azula_large"
    ["conditioned_navier_stokes"]="processor/conditioned_navier_stokes/crps_vit_azula_large"
    ["advection_diffusion"]="processor/advection_diffusion/crps_vit_azula_large"
)
declare -A AE_RUN_DIRS=(
    ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e"
    ["gpe_laser_only_wake"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f"
    ["conditioned_navier_stokes"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
    ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    ae_run_dir="${AE_RUN_DIRS[$datamodule]}"
    cache_dir="${ae_run_dir}/cached_latents"

    if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
        echo "Skipping ${datamodule}: cache missing train/valid/test under ${cache_dir}" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS-in-latent training"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  cache dir: ${cache_dir}"
        echo "  cosine_epochs: ${COSINE_EPOCHS}"

        uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
            local_experiment="${experiment}" \
            datamodule.data_path="${cache_dir}" \
            logging.wandb.enabled=true \
            optimizer.cosine_epochs="${COSINE_EPOCHS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${COSINE_EPOCHS}"
    done
done
