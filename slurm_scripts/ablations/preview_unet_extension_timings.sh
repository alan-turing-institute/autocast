#!/bin/bash
set -euo pipefail

# Preview only. This script never submits jobs. Run from the checkout root.
# Five epochs and the existing time-epochs workflow match the original timing
# procedure; original_unet_ablation is also the production callback policy.
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing_diffusion_unet_extensions}"
RUNS=(
    "unet_m8_crps_ad|advection_diffusion|advection_diffusion"
    "unet_m8_crps_gs|gray_scott|gray_scott"
    "unet_m8_crps_gpe|gpe_laser_only_wake|gpe_laser_wake_only"
)

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id dataset folder <<< "${run_spec}"
    uv run --frozen --no-sync autocast time-epochs \
        --kind epd --mode slurm --dry-run \
        --run-group "${RUN_GROUP}" --run-id "${run_id}" \
        -n 5 -b 24 -m 0.02 \
        local_experiment="ablations/arch_unet_fno_vit/${folder}/crps_unet_azula_80m" \
        trainer=original_unet_ablation \
        datamodule="${dataset}" \
        seed=42 \
        logging.wandb.name="${run_id}" \
        hydra.launcher.timeout_min=240
done
