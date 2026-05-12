#!/bin/bash

set -euo pipefail
# Eval submitter for the cross-dataset FairCRPS ablation runs.
#
# This mirrors the CRPS-ambient profile from submit_eval_planned_01.sh, but
# targets the FairCRPS m=8 ViT ablation for GS, GPE, and AD. The script can
# either auto-discover completed runs by W&B name under outputs/**/resolved_config.yaml
# or use explicit paths supplied via:
#
#   FAIR_CRPS_M8_GS_RUN_DIR=outputs/.../crps_gs64_...
#   FAIR_CRPS_M8_GPE_RUN_DIR=outputs/.../crps_gpe64_...
#   FAIR_CRPS_M8_AD_RUN_DIR=outputs/.../crps_ad64_...
#
# Output: <run>/eval_best_multiwinkler_from0p25/.

EVAL_BATCH_SIZE_CRPS=8
EVAL_N_MEMBERS=10
TIMEOUT_MIN_CRPS=240
EVAL_SUBDIR_MW="eval_best_multiwinkler_from0p25"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

# run_id|datamodule
RUNS=(
    "fair_crps_m8_gs|gray_scott"
    "fair_crps_m8_gpe|gpe_laser_only_wake"
    "fair_crps_m8_ad|advection_diffusion"
)

env_run_dir_var() {
    local run_id="$1"
    local upper
    upper="$(tr '[:lower:]' '[:upper:]' <<< "${run_id}")"
    printf '%s_RUN_DIR\n' "${upper}"
}

find_fair_crps_run_dir() {
    local run_id="$1"
    local env_var explicit_dir
    env_var="$(env_run_dir_var "${run_id}")"
    explicit_dir="${!env_var:-}"

    if [[ -n "${explicit_dir}" ]]; then
        printf '%s\n' "${explicit_dir}"
        return 0
    fi

    if [[ ! -d outputs ]]; then
        return 1
    fi

    local -a matches=()
    local cfg
    while IFS= read -r cfg; do
        if grep -q "name: ${run_id}$" "${cfg}" && \
           grep -q "autocast.losses.ensemble.FairCRPSLoss" "${cfg}"; then
            matches+=("$(dirname "${cfg}")")
        fi
    done < <(find outputs -path '*/resolved_config.yaml' ! -path '*/eval*/*' | sort)

    if (( ${#matches[@]} == 0 )); then
        return 1
    fi

    printf '%s\n' "${matches[$(( ${#matches[@]} - 1 ))]}"
}

resolve_multiwinkler_checkpoint() {
    local run_dir="$1"
    local -a ckpts=()

    mapfile -t ckpts < <(
        find "${run_dir}" -path '*/checkpoints/best-multiwinkler.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    mapfile -t ckpts < <(
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-overall-*.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    mapfile -t ckpts < <(
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-from0p25-*.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    return 1
}

submit_crps_ambient() {
    local run_id="$1" datamodule="$2" run_dir_abs="$3"
    local eval_ckpt
    if ! eval_ckpt="$(resolve_multiwinkler_checkpoint "${run_dir_abs}")"; then
        echo "Skipping ${run_id}: no best-multiwinkler checkpoint found" >&2
        return 0
    fi
    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR_MW}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        local dry_run_arg=() run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FairCRPS-ambient eval (best multi-Winkler)"
        echo "  mode: ${run_label}"
        echo "  run_id: ${run_id}"
        echo "  datamodule: ${datamodule}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR_MW}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR_MW}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE_CRPS}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN_CRPS}"
    done
}

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r run_id datamodule <<< "${run_spec}"

    if ! run_dir="$(find_fair_crps_run_dir "${run_id}")"; then
        env_var="$(env_run_dir_var "${run_id}")"
        echo "Skipping ${run_id}: no FairCRPS run found; set ${env_var}=outputs/..." >&2
        continue
    fi

    if [[ ! -d "${run_dir}" ]]; then
        echo "Skipping ${run_id}: run_dir missing at ${run_dir}" >&2
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_id}: resolved_config.yaml missing under ${run_dir}" >&2
        continue
    fi
    if ! grep -q "autocast.losses.ensemble.FairCRPSLoss" "${run_dir_abs}/resolved_config.yaml"; then
        echo "Skipping ${run_id}: run is not configured with FairCRPSLoss" >&2
        continue
    fi

    submit_crps_ambient "${run_id}" "${datamodule}" "${run_dir_abs}"
done
