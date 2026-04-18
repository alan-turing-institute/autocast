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
# Per-dataset cosine schedule: each (method, dataset) pair fills its own
# 24h budget so each model gets its best shot within budget. PLACEHOLDERS
# pending submit_crps_latent_timing.sh — extract per-dataset values via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# and replace each entry below.
#
# learning_rate (2e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
declare -A COSINE_EPOCHS_BY_DATASET=(
    ["gray_scott"]=1080                 # placeholder
    ["gpe_laser_only_wake"]=1080        # placeholder
    ["conditioned_navier_stokes"]=1080  # placeholder
    ["advection_diffusion"]=1080        # placeholder
)
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
    cosine_epochs="${COSINE_EPOCHS_BY_DATASET[$datamodule]}"
    # Save checkpoints at 25/50/75/100% of the schedule (top_k=-1 keeps all).
    # save_last: true (set in trainer/default.yaml) ensures last.ckpt captures
    # the final epoch even if it doesn't land on a quarter boundary.
    quarter_epochs=$((cosine_epochs / 4))

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
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast processor --mode slurm "${dry_run_arg[@]}" \
            local_experiment="${experiment}" \
            datamodule.data_path="${cache_dir}" \
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
