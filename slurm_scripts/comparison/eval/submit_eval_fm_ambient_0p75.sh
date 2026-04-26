#!/bin/bash

set -euo pipefail
# Evaluate FM cached-latent processor runs from 2026-04-20 at the
# 75%-progress checkpoint with explicit eval.mode=ambient.
#
# This charges the autoencoder decode->encode drift at every rollout step.
# Outputs are kept under eval_0p75_ambient/ so they do not overwrite the
# default auto->encode_once 75%-checkpoint evals.
#
# Current 2026-04-20 FM runs saved legacy quarter-*.ckpt files, so this uses
# the third sorted quarter checkpoint as the 75% fallback. Future runs with
# snapshot-0p75-*.ckpt files are preferred automatically.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
EVAL_SUBDIR="eval_0p75_ambient"
PROGRESS_TOKEN="0p75"
PROGRESS_LABEL="0.75"
LEGACY_QUARTER_INDEX=2
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"
    "outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"
    "outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"
    "outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
)

resolve_progress_checkpoint() {
    local run_dir="$1"
    local progress_token="$2"
    local legacy_quarter_index="$3"
    local -a snapshot_ckpts=()
    local -a quarter_ckpts=()

    mapfile -t snapshot_ckpts < <(
        find "${run_dir}" -type f -path "*/checkpoints/snapshot-${progress_token}-*.ckpt" | sort
    )

    if (( ${#snapshot_ckpts[@]} >= 1 )); then
        printf '%s\n' "${snapshot_ckpts[$(( ${#snapshot_ckpts[@]} - 1 ))]}"
        return 0
    fi

    mapfile -t quarter_ckpts < <(
        find "${run_dir}" -type f -path '*/checkpoints/quarter-*.ckpt' | sort
    )

    if (( ${#quarter_ckpts[@]} > legacy_quarter_index )); then
        printf '%s\n' "${quarter_ckpts[$legacy_quarter_index]}"
        return 0
    fi

    return 1
}

for run_dir in "${RUN_DIRS[@]}"; do
    ae_ckpt="${AE_CKPT[$run_dir]:-}"
    if [[ -z "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: no autoencoder_checkpoint mapping" >&2
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: AE checkpoint missing at ${ae_ckpt}" >&2
        continue
    fi
    ae_ckpt_abs="$(realpath "${ae_ckpt}")"

    if ! eval_ckpt="$(resolve_progress_checkpoint "${run_dir_abs}" "${PROGRESS_TOKEN}" "${LEGACY_QUARTER_INDEX}")"; then
        echo "Skipping ${run_dir}: neither snapshot-${PROGRESS_TOKEN}-*.ckpt nor legacy quarter checkpoint index ${LEGACY_QUARTER_INDEX} found" >&2
        continue
    fi
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FM ambient eval (${PROGRESS_LABEL} checkpoint)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            +autoencoder_checkpoint="${ae_ckpt_abs}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
