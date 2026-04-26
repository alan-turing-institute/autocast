#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../comparison/cached_latents/validate_cached_latents_against_ae.sh"

# 24h production jobs for planned ablation batch 02.
#
# Run submit_planned_02_timing.sh first. This script resolves each
# trainer.max_epochs from either COSINE_EPOCHS_BY_RUN or the latest matching
# timing.ckpt under outputs/*/timing_planned_02/<run_id>/timing.ckpt.

declare -A COSINE_EPOCHS_BY_RUN=(
    # Optional pinned values, e.g. after timing:
    # ["latent_crps_m8_gs"]=123
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
RUN_GROUP="$(date +%Y-%m-%d)/planned_02"
RUN_DRY_STATES=("true" "false")

declare -A AE_RUN_DIRS=(
    ["gray_scott"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e"
    ["gpe_laser_only_wake"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f"
    ["advection_diffusion"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300"
)

# run_id|kind|datamodule|local_experiment
RUNS=(
    "latent_crps_m8_gs|processor|gray_scott|processor/gray_scott/crps_vit_azula_large"
    "latent_crps_m8_gpe|processor|gpe_laser_only_wake|processor/gpe_laser_wake_only/crps_vit_azula_large"
    "latent_crps_m8_ad|processor|advection_diffusion|processor/advection_diffusion/crps_vit_azula_large"
    "vit_m4_gs|epd|gray_scott|epd/gray_scott/crps_vit_azula_large"
)

run_overrides() {
    local run_id="$1"
    local kind="$2"
    local datamodule="$3"

    case "${kind}" in
        processor)
            local ae_run_dir="${AE_RUN_DIRS[$datamodule]}"
            local cache_dir="${ae_run_dir}/cached_latents"
            printf '%s\n' "datamodule=cached_latents" "datamodule.data_path=${cache_dir}"
            ;;
        epd)
            case "${run_id}" in
                vit_m4_gs)
                    printf '%s\n' "datamodule=${datamodule}" "model.n_members=4" "datamodule.batch_size=64"
                    ;;
                *)
                    printf '%s\n' "datamodule=${datamodule}"
                    ;;
            esac
            ;;
        *)
            echo "FATAL: unknown kind '${kind}' for ${run_id}" >&2
            exit 1
            ;;
    esac
}

validate_run_inputs() {
    local run_id="$1"
    local kind="$2"
    local datamodule="$3"

    if [[ "${kind}" != "processor" ]]; then
        return 0
    fi

    local ae_run_dir="${AE_RUN_DIRS[$datamodule]:-}"
    if [[ -z "${ae_run_dir}" ]]; then
        echo "Skipping ${run_id}: no AE run dir configured for ${datamodule}" >&2
        return 1
    fi

    local cache_dir="${ae_run_dir}/cached_latents"
    if [[ ! -d "${cache_dir}/train" ]] || [[ ! -d "${cache_dir}/valid" ]] || [[ ! -d "${cache_dir}/test" ]]; then
        echo "Skipping ${run_id}: cache missing train/valid/test under ${cache_dir}" >&2
        return 1
    fi
    if ! validate_cached_latents_against_ae "${ae_run_dir}"; then
        echo "Skipping ${run_id}: cached-latents config mismatch vs AE training config" >&2
        return 1
    fi
}

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/timing_planned_02/${run_id}/timing.ckpt" | sort | tail -n 1
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
    IFS="|" read -r run_id kind datamodule experiment <<< "${run_spec}"
    if ! validate_run_inputs "${run_id}" "${kind}" "${datamodule}"; then
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

    mapfile -t overrides < <(run_overrides "${run_id}" "${kind}" "${datamodule}")

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting planned batch 02 production run"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  kind: ${kind}"
        echo "  datamodule: ${datamodule}"
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
