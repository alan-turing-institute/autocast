#!/bin/bash

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/validate_cached_latents_against_ae.sh"
# Submit FM-in-latent timing jobs for 4 target datasets.
# Model: flow_matching_vit (vit backbone, hid_channels=704, hid_blocks=12,
# attention_heads=8, patch_size=1, flow_ode_steps=50). Optimizer: adamw_half
# (LR=1e-4, warmup=0). Batch size: 256/GPU.
# See local_hydra/local_experiment/processor/<dataset>/fm_vit_large.yaml for
# the authoritative hyperparameters.
# Runs 5 epochs each to measure per-epoch wall-clock time, then use
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# to compute the max_epochs for a 24h budget.

# Per-dataset local_experiment + AE run dir (cached latents live under
# <ae_run_dir>/cached_latents/). Fill in paths once submit_cache_latents.sh
# has completed.
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

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    ae_run_dir="${AE_RUN_DIRS[$datamodule]}"
    cache_dir="${ae_run_dir}/cached_latents"

    if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
        echo "Skipping ${datamodule}: cache missing train/valid/test under ${cache_dir}" >&2
        continue
    fi
    if ! validate_cached_latents_against_ae "${ae_run_dir}"; then
        echo "Skipping ${datamodule}: cached-latents config mismatch vs AE training config" >&2
        continue
    fi

    echo "Submitting FM-in-latent timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  cache dir: ${cache_dir}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind processor --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "fm_b256_${datamodule}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}" \
        datamodule.data_path="${cache_dir}"

    echo ""
    echo "---"
    echo ""
done

echo "All timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/fm_b256_*/retrieve.sh; do bash \"\$f\"; done"
