#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/validate_cached_latents_against_ae.sh"
# Final 24h FM-in-latent runs for 4 target datasets.
# Model: flow_matching_vit (vit backbone, hid_channels=704, hid_blocks=12,
# attention_heads=8, patch_size=1, flow_ode_steps=50). Optimizer: adamw_half
# (LR=1e-4, warmup=0). Batch size: 256/GPU.
# See local_hydra/local_experiment/processor/<dataset>/fm_vit_large.yaml for
# the authoritative hyperparameters.
#
# Per-dataset cosine schedule: each (method, dataset) pair fills its own
# 24h budget so each model gets its best shot within budget. Values from
# submit_fm_timing.sh (2026-04-18) via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
#
# learning_rate (1e-4) and warmup (0) are baked into each per-dataset
# local_experiment config; adjust the yaml to change them.
declare -A COSINE_EPOCHS_BY_DATASET=(
    ["gray_scott"]=2830                 # 29.9 s/ep
    ["gpe_laser_only_wake"]=3411        # 24.8 s/ep
    ["conditioned_navier_stokes"]=3223  # 26.3 s/ep
    ["advection_diffusion"]=3314        # 25.6 s/ep
)
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

# Per-dataset local_experiment + AE run dir (cached latents live under
# <ae_run_dir>/cached_latents/).
declare -A EXPERIMENTS=(
    ["gray_scott"]="processor/gray_scott/fm_vit_large"
    ["gpe_laser_only_wake"]="processor/gpe_laser_wake_only/fm_vit_large"
    ["conditioned_navier_stokes"]="processor/conditioned_navier_stokes/fm_vit_large"
    ["advection_diffusion"]="processor/advection_diffusion/fm_vit_large"
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
    # Save checkpoints every ~5% of optimizer-step progress (top_k=-1 keeps all).
    # save_last: true (set in trainer/default.yaml) ensures last.ckpt captures
    # the final state even if it doesn't land exactly on a progress boundary.

    if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
        echo "Skipping ${datamodule}: cache missing train/valid/test under ${cache_dir}" >&2
        continue
    fi
    if ! validate_cached_latents_against_ae "${ae_run_dir}"; then
        echo "Skipping ${datamodule}: cached-latents config mismatch vs AE training config" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FM-in-latent training"
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
            trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
            trainer.callbacks.0.every_n_epochs=0 \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
    done
done
