#!/bin/bash

set -euo pipefail
# Cache autoencoder latents for the 4 target datasets.
# Each entry pairs a local_experiment config (which bakes in the datamodule
# + encoder/decoder architecture — periodic, pixel_shuffle — matching the
# per-dataset AE under local_hydra/local_experiment/ae/<dataset>/) with the
# AE run dir whose checkpoint will be loaded.
#
# Checkpoint lookup: prefers <ae_run_dir>/autoencoder.ckpt (written at the
# end of a full AE run). If that doesn't exist (AE still training) the
# newest <ae_run_dir>/autocast/*/checkpoints/latest-*.ckpt is used instead —
# fine for timing since compute throughput doesn't depend on AE quality.
# Do NOT use a temp ckpt for the final `large/` runs — re-run this script
# once submit_ae_large.sh has completed and autoencoder.ckpt is in place.
#
# Latents are written to <ae_run_dir>/cached_latents/{train,valid,test}.
# Normalization must match AE training. We read use_normalization from
# <ae_run_dir>/resolved_autoencoder_config.yaml when available and pass it
# explicitly to cache-latents to avoid drifting datamodule defaults.
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
    ae_resolved_cfg="${ae_run_dir}/resolved_autoencoder_config.yaml"

    ckpt="${ae_run_dir}/autoencoder.ckpt"

    if [[ ! -f "$ckpt" ]]; then
        # Fallback: newest latest-*.ckpt from any wandb run under this ae_run_dir.
        ckpt="$(ls -t "${ae_run_dir}"/autocast/*/checkpoints/latest-*.ckpt 2>/dev/null | head -n 1 || true)"
        if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
            echo "Skipping ${datamodule}: no autoencoder.ckpt or latest-*.ckpt under ${ae_run_dir}" >&2
            continue
        fi
        echo "Using temp checkpoint (AE still training?): ${ckpt}" >&2
    fi

    cache_workdir="${ae_run_dir}/cached_latents"
    if [[ ! -f "${ae_resolved_cfg}" ]]; then
        echo "Skipping ${datamodule}: missing ${ae_resolved_cfg} (can't safely infer normalization)" >&2
        continue
    fi
    ae_use_normalization="$(awk '/^[[:space:]]*use_normalization:[[:space:]]*/ {print $2; exit}' "${ae_resolved_cfg}")"
    if [[ "${ae_use_normalization}" != "true" && "${ae_use_normalization}" != "false" ]]; then
        echo "Skipping ${datamodule}: could not parse use_normalization from ${ae_resolved_cfg}" >&2
        continue
    fi

    if [[ -d "$cache_workdir" ]]; then
        echo "Warning: cache workdir already exists, will overwrite: $cache_workdir" >&2
    fi

    echo "Submitting latent cache job"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  autoencoder run: ${ae_run_dir}"
    echo "  datamodule.use_normalization: ${ae_use_normalization}"
    echo "  cache workdir: ${cache_workdir}"

    uv run autocast cache-latents --mode slurm \
        --workdir "${cache_workdir}" \
        --output-dir "${cache_workdir}" \
        autoencoder_checkpoint="${ckpt}" \
        local_experiment="${experiment}" \
        datamodule.use_normalization="${ae_use_normalization}" \
        hydra.launcher.timeout_min=120 || echo "FAILED to submit: ${datamodule}" >&2
done
