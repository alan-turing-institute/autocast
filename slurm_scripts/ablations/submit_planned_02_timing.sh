#!/bin/bash

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../comparison/cached_latents/validate_cached_latents_against_ae.sh"

# Timing jobs for planned ablation batch 02.
#
# This deliberately sits next to, rather than edits,
# submit_planned_01_timing.sh so the first CNS batch can remain a distinct
# submission set.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_02"

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

for run_spec in "${RUNS[@]}"; do
    IFS="|" read -r run_id kind datamodule experiment <<< "${run_spec}"
    if ! validate_run_inputs "${run_id}" "${kind}" "${datamodule}"; then
        continue
    fi

    mapfile -t overrides < <(run_overrides "${run_id}" "${kind}" "${datamodule}")

    echo "Submitting planned batch 02 timing run"
    echo "  run_id: ${run_id}"
    echo "  kind: ${kind}"
    echo "  datamodule: ${datamodule}"
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

echo "All planned batch 02 timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/*/retrieve.sh; do bash \"\$f\"; done"
