#!/bin/bash

set -euo pipefail
# Submit primary CRPS-via-AE timing jobs for 4 target datasets.
# This variant uses EPD with a pretrained autoencoder checkpoint for
# encode/decode around the processor, while CRPS is still computed in ambient
# space (train_in_latent_space=false).
#
# FM cached-latent runs remain primary for FM.
# CRPS cached-latent runs are retained separately as ablations.

declare -A EXPERIMENTS=(
    ["gray_scott"]="epd/gray_scott/crps_vit_azula_large_ae_ambient"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large_ae_ambient"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large_ae_ambient"
    ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large_ae_ambient"
)
declare -A AE_RUN_DIRS=(
    ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e"
    ["gpe_laser_only_wake"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f"
    ["conditioned_navier_stokes"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
    ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    ae_run_dir="${AE_RUN_DIRS[$datamodule]}"

    ckpt="${ae_run_dir}/autoencoder.ckpt"
    if [[ ! -f "${ckpt}" ]]; then
        ckpt="$(ls -t "${ae_run_dir}"/autocast/*/checkpoints/latest-*.ckpt 2>/dev/null | head -n 1 || true)"
        if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
            echo "Skipping ${datamodule}: no autoencoder.ckpt or latest-*.ckpt under ${ae_run_dir}" >&2
            continue
        fi
        echo "Using temp checkpoint (AE still training?): ${ckpt}" >&2
    fi

    echo "Submitting CRPS-via-AE (ambient loss) timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  autoencoder checkpoint: ${ckpt}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "crps_ae_ambient_b32_${datamodule}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}" \
        autoencoder_checkpoint="${ckpt}"

    echo ""
    echo "---"
    echo ""
done

echo "All timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/crps_ae_ambient_b32_*/retrieve.sh; do bash \"\$f\"; done"