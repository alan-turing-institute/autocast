#!/bin/bash

set -euo pipefail
# Ambient eval submitter for the 75%-progress checkpoints from the current
# CRPS ensemble-size runs.
#
# This mirrors ../eval/submit_eval_crps_ambient.sh but keeps outputs under a
# sibling eval_0p75/ folder so partial-schedule metrics, rollout videos, and
# SLURM logs do not mix with the standard final-checkpoint evals.
#
# Current large-run training scripts save checkpoints every 5% of training
# progress via snapshot-<progress>-*.ckpt filenames. For older quarter-schedule
# runs, fall back to the third sorted quarter-*.ckpt checkpoint.
#
# Force eval.mode=ambient here. These intermediate checkpoints can look
# processor-only to the early eval.mode=auto dispatcher because stateless
# encoders/decoders (PermuteConcat / ChannelsLast) contribute no
# encoder_decoder.* weights, even though the full raw-space ambient path is the
# correct evaluation route for these runs.
#
# Batch size: keep 4/GPU as the same conservative first pass used by the
# standard ambient ensemble-size eval script.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
EVAL_SUBDIR="eval_0p75"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    # CNS pilot runs retained alongside the compute-matched sweep.
    # "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_5e157a5"  # ensemble_m16_fixed_bs32
    # Compute-matched eff_bs1024 runs across comparison datasets.
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_dcd79e4"
    "outputs/2026-04-21/ensemble_size/crps_gs64_vit_azula_large_ac1bb06_639963f"
    "outputs/2026-04-21/ensemble_size/crps_gpe64_vit_azula_large_ac1bb06_638585e"
    "outputs/2026-04-21/ensemble_size/crps_ad64_vit_azula_large_ac1bb06_ef6368d"
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
    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi

    if ! eval_ckpt="$(resolve_progress_checkpoint "${run_dir_abs}" "0p75" 2)"; then
        echo "Skipping ${run_dir}: neither snapshot-0p75-*.ckpt nor legacy third quarter-*.ckpt found" >&2
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

        echo "Submitting ensemble-size CRPS ambient eval (0.75 checkpoint)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
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
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
