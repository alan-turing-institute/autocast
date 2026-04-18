#!/bin/bash

set -euo pipefail
# Final 24h primary CRPS-via-AE runs for 4 target datasets.
# This variant uses EPD with a pretrained autoencoder checkpoint for
# encode/decode around the processor, while CRPS is still computed in ambient
# space (train_in_latent_space=false).
#
# Replace the placeholder COSINE_EPOCHS values after running
# submit_crps_ae_ambient_timing.sh and extracting:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24

declare -A COSINE_EPOCHS_BY_DATASET=(
    # ["gray_scott"]=49                   # 1724.6 s/ep (timing_efficient_crps, 2026-04-18)
    # ["gpe_laser_only_wake"]=85          #  991.0 s/ep (timing_efficient_crps, 2026-04-18)
    ["conditioned_navier_stokes"]=85    #  985.0 s/ep (timing_efficient_crps, 2026-04-18)
    # ["advection_diffusion"]=58          # 1436.9 s/ep (timing_efficient_crps, 2026-04-18)
)
BUDGET_MAX_TIME="00:23:59:00"
# SLURM timeout with 1-min buffer beyond the 24h budget.
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

declare -A EXPERIMENTS=(
    # ["gray_scott"]="epd/gray_scott/crps_vit_azula_large_ae_ambient"
    # ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large_ae_ambient"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large_ae_ambient"
    # ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large_ae_ambient"
)
declare -A AE_RUN_DIRS=(
    # ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e"
    # ["gpe_laser_only_wake"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f"
    ["conditioned_navier_stokes"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
    # ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300"
)

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    ae_run_dir="${AE_RUN_DIRS[$datamodule]}"
    cosine_epochs="${COSINE_EPOCHS_BY_DATASET[$datamodule]}"

    ckpt="${ae_run_dir}/autoencoder.ckpt"
    if [[ ! -f "${ckpt}" ]]; then
        ckpt="$(ls -t "${ae_run_dir}"/autocast/*/checkpoints/latest-*.ckpt 2>/dev/null | head -n 1 || true)"
        if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
            echo "Skipping ${datamodule}: no autoencoder.ckpt or latest-*.ckpt under ${ae_run_dir}" >&2
            continue
        fi
        echo "Using temp checkpoint (AE still training?): ${ckpt}" >&2
    fi

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

        echo "Submitting CRPS-via-AE (ambient loss) training"
        echo "  mode: ${run_label}"
        echo "  datamodule: ${datamodule}"
        echo "  local_experiment: ${experiment}"
        echo "  autoencoder checkpoint: ${ckpt}"
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
            local_experiment="${experiment}" \
            autoencoder_checkpoint="${ckpt}" \
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