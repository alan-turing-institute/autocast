#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../comparison/cached_latents/validate_cached_latents_against_ae.sh"

# Timing jobs for planned ablation batch 01.
#
# This is intentionally CNS-only and cross-cuts the study-specific ablation
# folders. Architecture-changing variants use local_experiment configs under
# local_hydra/local_experiment/ablations/*; one-knob variants use CLI
# overrides against the canonical comparison configs.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_01"
SOURCE_DATASET="conditioned_navier_stokes"
AE_RUN_DIR="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8"
CACHE_DIR="${AE_RUN_DIR}/cached_latents"

# run_id|kind|local_experiment
RUNS=(
    "unet_m8_crps_cns|epd|ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_unet_azula_80m"
    "diffusion_cns|epd|ablations/fm_vs_diffusion/conditioned_navier_stokes/diffusion_vit_large"
    "fair_crps_m8_cns|epd|ablations/crps_variants/conditioned_navier_stokes/crps_vit_fair"
    "plain_crps_m8_cns|epd|ablations/crps_variants/conditioned_navier_stokes/crps_vit_plain"
    "vit_noise256_m8_cns|epd|ablations/noise_channels/conditioned_navier_stokes/crps_vit_noise256"
    "vit_m4_cns|epd|epd/conditioned_navier_stokes/crps_vit_azula_large"
    "latent_crps_m8_cns|processor|processor/conditioned_navier_stokes/crps_vit_azula_large"
    "vit_global_cond_m8_cns|epd|epd/conditioned_navier_stokes/crps_vit_azula_large_identity_global_cond"
)

run_overrides() {
    local run_id="$1"

    case "${run_id}" in
        latent_crps_m8_cns)
            # Keep the local_experiment's cached_latents datamodule. Passing
            # datamodule=conditioned_navier_stokes here would switch training
            # back to raw fields. Pair datamodule=cached_latents with the cache
            # path so the workflow infers the source cns64 token from
            # autoencoder_config.yaml.
            printf '%s\n' "datamodule=cached_latents" "datamodule.data_path=${CACHE_DIR}"
            ;;
        vit_m4_cns)
            printf '%s\n' "datamodule=${SOURCE_DATASET}" "model.n_members=4" "datamodule.batch_size=64"
            ;;
        *)
            printf '%s\n' "datamodule=${SOURCE_DATASET}"
            ;;
    esac
}

validate_run_inputs() {
    local run_id="$1"

    if [[ "${run_id}" != "latent_crps_m8_cns" ]]; then
        return 0
    fi

    if [[ ! -d "${CACHE_DIR}/train" ]] || [[ ! -d "${CACHE_DIR}/valid" ]] || [[ ! -d "${CACHE_DIR}/test" ]]; then
        echo "Skipping ${run_id}: cache missing train/valid/test under ${CACHE_DIR}" >&2
        return 1
    fi
    if ! validate_cached_latents_against_ae "${AE_RUN_DIR}"; then
        echo "Skipping ${run_id}: cached-latents config mismatch vs AE training config" >&2
        return 1
    fi
}

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id kind experiment <<< "${run_spec}"
    if ! validate_run_inputs "${run_id}"; then
        continue
    fi

    mapfile -t overrides < <(run_overrides "${run_id}")

    echo "Submitting planned batch 01 timing run"
    echo "  run_id: ${run_id}"
    echo "  kind: ${kind}"
    echo "  source_dataset: ${SOURCE_DATASET}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"

    uv run autocast time-epochs --kind "${kind}" --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}" \
        "${overrides[@]}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned batch 01 timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/*/retrieve.sh; do bash \"\$f\"; done"
