#!/bin/bash

set -euo pipefail
# Snapshots variant of submit_eval_crps_ambient_best_multiwinkler_from0p25.sh.
#
# Writes outputs to a sibling subdir suffixed `_snapshots` so the figure-only
# re-run lives next to the original eval. Supports running each eval either via
# SLURM (default, sbatch through the autocast workflow) or locally on the
# current machine.
#
# Usage:
#   ./submit_eval_crps_ambient_best_multiwinkler_from0p25_snapshots.sh           # slurm (default)
#   RUN_MODE=local ./submit_eval_crps_ambient_best_multiwinkler_from0p25_snapshots.sh
#   RUN_MODE=slurm ./submit_eval_crps_ambient_best_multiwinkler_from0p25_snapshots.sh
#
# In slurm mode both real and `--dry-run` submissions are performed, matching
# the parent script. In local mode only the real run is executed (dry-run is a
# slurm-submission concern).

EVAL_BATCH_SIZE=8
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
EVAL_SUBDIR="eval_best_multiwinkler_from0p25_snapshots"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
RUN_MODE="${RUN_MODE:-slurm}"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

case "${RUN_MODE}" in
    slurm) RUN_DRY_STATES=("true" "false") ;;
    local) RUN_DRY_STATES=("false") ;;
    *)
        echo "Unknown RUN_MODE='${RUN_MODE}'. Expected 'slurm' or 'local'." >&2
        exit 2
        ;;
esac

RUN_DIRS=(
    "outputs/2026-04-24/crps_gs64_vit_azula_large_bed4611_828a161"
    "outputs/2026-04-24/crps_gpe64_vit_azula_large_bed4611_e0a6df5"
    "outputs/2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534"
    "outputs/2026-04-24/crps_ad64_vit_azula_large_bed4611_da01a04"
)

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
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-from0p25-*.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
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

    if ! eval_ckpt="$(resolve_multiwinkler_checkpoint "${run_dir_abs}")"; then
        echo "Skipping ${run_dir}: no best-multiwinkler.ckpt symlink and no best-multiwinkler-from0p25-*.ckpt" >&2
        continue
    fi
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="${RUN_MODE}"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="${RUN_MODE} --dry-run"
        fi

        echo "Submitting CRPS-ambient snapshots eval (best multi-Winkler from 0.25)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.rollout_snapshot_timesteps: ${ROLLOUT_SNAPSHOT_TIMESTEPS}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        cmd=(uv run autocast eval --mode "${RUN_MODE}" "${dry_run_arg[@]}"
            --workdir "${run_dir_abs}"
            --output-subdir "${EVAL_SUBDIR}"
            eval.checkpoint="${eval_ckpt_abs}"
            eval.mode=ambient
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv"
            eval.video_dir="${eval_output_dir}/videos"
            eval.save_rollout_snapshots=true
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots"
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}"
            eval.rollout_snapshot_format=png
            eval.metrics="${EVAL_METRICS}"
            eval.batch_size="${EVAL_BATCH_SIZE}"
            eval.n_members="${EVAL_N_MEMBERS}")

        if [[ "${RUN_MODE}" == "slurm" ]]; then
            cmd+=(hydra.launcher.timeout_min="${TIMEOUT_MIN}")
        fi

        "${cmd[@]}"
    done
done
