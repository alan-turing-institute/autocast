#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../comparison/cached_latents/validate_cached_latents_against_ae.sh"

# 24h production jobs for the planned CNS ablation batch.
#
# Run submit_planned_cns_timing.sh first. This script resolves each
# trainer.max_epochs from either COSINE_EPOCHS_BY_RUN or the latest matching
# timing.ckpt under outputs/*/timing_planned_cns/<run_id>/timing.ckpt.

declare -A COSINE_EPOCHS_BY_RUN=(
    # Pinned from outputs/2026-04-25/timing_planned_cns/*/timing.ckpt at
    # 24h budget, 2% margin (uv run autocast time-epochs -b 24 -m 0.02).
    ["unet_m8_crps_cns"]=611
    ["diffusion_cns"]=2248
    ["fair_crps_m8_cns"]=439
    ["plain_crps_m8_cns"]=439
    ["vit_noise256_m8_cns"]=427
    ["vit_m4_cns"]=803
    ["latent_crps_m8_cns"]=327
    ["vit_global_cond_m8_cns"]=432
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
SOURCE_DATASET="conditioned_navier_stokes"
RUN_GROUP="$(date +%Y-%m-%d)/planned_cns"
RUN_DRY_STATES=("true" "false")
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

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/timing_planned_cns/${run_id}/timing.ckpt" | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs --from-checkpoint "${timing_ckpt}" -b 24 -m 0.02
    )"

    sed -n 's/.*trainer.max_epochs=\([0-9][0-9]*\).*/\1/p' <<< "${result}" | tail -n 1
}

resolve_cosine_epochs() {
    local run_id="$1"
    local cached="${COSINE_EPOCHS_BY_RUN[$run_id]:-}"

    if [[ -n "${cached}" ]]; then
        printf '%s\n' "${cached}"
        return 0
    fi

    local timing_ckpt
    timing_ckpt="$(find_timing_checkpoint "${run_id}")"
    if [[ -z "${timing_ckpt}" ]]; then
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id kind experiment <<< "${run_spec}"
    if ! validate_run_inputs "${run_id}"; then
        continue
    fi

    if ! cosine_epochs="$(resolve_cosine_epochs "${run_id}")"; then
        echo "Skipping ${run_id}: no timing-derived cosine_epochs available" >&2
        continue
    fi
    if [[ -z "${cosine_epochs}" ]]; then
        echo "Skipping ${run_id}: could not parse trainer.max_epochs from timing output" >&2
        continue
    fi

    mapfile -t overrides < <(run_overrides "${run_id}")

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting planned CNS production run"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  kind: ${kind}"
        echo "  source_dataset: ${SOURCE_DATASET}"
        echo "  local_experiment: ${experiment}"
        echo "  cosine_epochs: ${cosine_epochs}"

        uv run autocast "${kind}" --mode slurm "${dry_run_arg[@]}" \
            --run-group "${RUN_GROUP}" \
            local_experiment="${experiment}" \
            "${overrides[@]}" \
            logging.wandb.enabled=true \
            logging.wandb.name="${run_id}" \
            optimizer.cosine_epochs="${cosine_epochs}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
            trainer.max_time="${BUDGET_MAX_TIME}" \
            +trainer.max_epochs="${cosine_epochs}" \
            trainer.callbacks.0.every_n_train_steps_fraction=0.05 \
            +trainer.callbacks.0.every_n_epochs=0 \
            trainer.callbacks.0.save_top_k=-1 \
            trainer.callbacks.0.filename=\"snapshot-{progress_token}-{epoch:04d}-{step:08d}\"
    done
done
