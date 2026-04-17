#!/bin/bash

set -euo pipefail
# Cache autoencoder latents for the 4 target datasets.
# Each entry pairs a local_experiment config (which bakes in the datamodule
# + encoder/decoder architecture — periodic, pixel_shuffle — matching the
# per-dataset AE under local_hydra/local_experiment/ae/<dataset>/) with the
# AE run dir whose checkpoint will be loaded.
#
# The checkpoint is read from <ae_run_dir>/autoencoder.ckpt and latents are
# written to <ae_run_dir>/cached_latents/{train,valid,test}.
#
# Fill in each ae_run_dir once submit_ae_large.sh completes — the paths below
# are placeholders matching the 2026-04-17 launch (still training at time of
# writing).
declare -A EXPERIMENTS=(
    ["gray_scott"]="cache_latents/gray_scott/cache_latents"
    ["gpe_laser_only_wake"]="cache_latents/gpe_laser_wake_only/cache_latents"
    ["conditioned_navier_stokes"]="cache_latents/conditioned_navier_stokes/cache_latents"
    ["advection_diffusion"]="cache_latents/advection_diffusion/cache_latents"
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

    ckpt="${ae_run_dir}/autoencoder.ckpt"

    if [[ ! -f "$ckpt" ]]; then
        echo "Skipping ${datamodule}: missing checkpoint ${ckpt}" >&2
        continue
    fi

    cache_workdir="${ae_run_dir}/cached_latents"

    if [[ -d "$cache_workdir" ]]; then
        echo "Warning: cache workdir already exists, will overwrite: $cache_workdir" >&2
    fi

    echo "Submitting latent cache job"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  autoencoder run: ${ae_run_dir}"
    echo "  cache workdir: ${cache_workdir}"

    uv run autocast cache-latents --mode slurm \
        --workdir "${cache_workdir}" \
        --output-dir "${cache_workdir}" \
        autoencoder_checkpoint="${ckpt}" \
        local_experiment="${experiment}" \
        hydra.launcher.timeout_min=120 || echo "FAILED to submit: ${datamodule}" >&2
done
