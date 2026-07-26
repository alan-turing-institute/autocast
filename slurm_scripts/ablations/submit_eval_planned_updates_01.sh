#!/bin/bash

set -euo pipefail

# Submit the four planned_updates_01 MC-dropout CRPS evaluations after their
# corresponding training jobs leave the queue.
#
# The overall-best multi-Winkler checkpoint is resolved inside the allocated
# evaluation job, after its afterany dependency is satisfied. Resolution is
# strict: exactly one best-multiwinkler-overall-*.ckpt must exist.
#
# Preview all four submissions without queueing anything:
#   ./slurm_scripts/ablations/submit_eval_planned_updates_01.sh
#
# Queue them:
#   SUBMIT=true ./slurm_scripts/ablations/submit_eval_planned_updates_01.sh

EVAL_BATCH_SIZE=8
EVAL_N_MEMBERS=10
# The matching ambient CRPS evaluations completed in 4-11 minutes. Retain the
# same conservative limit selected for the seed-43 comparison evaluations.
TIMEOUT_MIN=45
MEMORY="115G"
EVAL_SUBDIR="eval_best_multiwinkler_overall"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

run_deferred_eval() {
    local repo_root="$1"
    local run_dir="$2"
    local -a checkpoints=()

    cd "${repo_root}"

    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing resolved config: ${run_dir}/resolved_config.yaml" >&2
        return 1
    fi

    mapfile -t checkpoints < <(
        find "${run_dir}" -type f \
            -path '*/checkpoints/best-multiwinkler-overall-*.ckpt' \
            -print | sort
    )

    if (( ${#checkpoints[@]} != 1 )); then
        echo "Expected exactly one best-multiwinkler-overall checkpoint in ${run_dir}; found ${#checkpoints[@]}" >&2
        if (( ${#checkpoints[@]} > 0 )); then
            printf '  %s\n' "${checkpoints[@]}" >&2
        fi
        return 1
    fi

    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${checkpoints[0]}")"
    eval_output_dir="${run_dir}/${EVAL_SUBDIR}"

    echo "Starting deferred MC-dropout CRPS ambient evaluation"
    echo "  run_dir: ${run_dir}"
    echo "  eval.checkpoint: ${eval_ckpt_abs}"
    echo "  eval.mode: ambient"
    echo "  output_subdir: ${EVAL_SUBDIR}"
    echo "  time: ${TIMEOUT_MIN} minutes"
    echo "  memory: ${MEMORY}"

    srun --nodes=1 --ntasks=1 --gpus=1 \
        uv run autocast eval --mode local \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_format=png \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            eval.devices=1
}

if [[ "${1:-}" == "--run-deferred" ]]; then
    if (( $# != 3 )); then
        echo "Usage: $0 --run-deferred REPO_ROOT RUN_DIR" >&2
        exit 2
    fi
    run_deferred_eval "$2" "$3"
    exit
fi

RUN_ROOT="${RUN_ROOT:-outputs/2026-07-25/planned_updates_01}"
SUBMIT="${SUBMIT:-false}"

case "${SUBMIT}" in
    true|false) ;;
    *)
        echo "SUBMIT must be true or false, got: ${SUBMIT}" >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
script_path="$(realpath "${BASH_SOURCE[0]}")"

# run_id|training_job_id|run_directory
RUNS=(
    "mcdo_ad|5776844|epd_ad64_vit_azula_mc_dropout_large_f008bcf_df1b2b5"
    "mcdo_gs|5776845|epd_gs64_vit_azula_mc_dropout_large_f008bcf_6784acb"
    "mcdo_gpe|5776846|epd_gpe64_vit_azula_mc_dropout_large_f008bcf_299ad0f"
    "mcdo_cns|5776847|epd_cns64_vit_azula_mc_dropout_large_f008bcf_8ee53a1"
)

had_error=false

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r run_id training_job_id run_name <<< "${run_spec}"
    run_dir="${RUN_ROOT%/}/${run_name}"

    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing resolved config for ${run_id}: ${run_dir}" >&2
        had_error=true
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    echo "plan: ${run_id}"
    echo "  training dependency: afterany:${training_job_id}"
    echo "  run_dir: ${run_dir_abs}"
    echo "  deferred checkpoint: best-multiwinkler-overall-*.ckpt"
    echo "  eval.mode: ambient"
    echo "  output_subdir: ${EVAL_SUBDIR}"
    echo "  time: ${TIMEOUT_MIN} minutes"
    echo "  memory: ${MEMORY}"

    if [[ "${SUBMIT}" == "false" ]]; then
        continue
    fi

    mkdir -p "${eval_output_dir}"
    eval_job_id="$(
        sbatch --parsable \
            --job-name="eval_${run_id}_overall" \
            --output="${eval_output_dir}/slurm-%j.out" \
            --error="${eval_output_dir}/slurm-%j.err" \
            --time="${TIMEOUT_MIN}" \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --mem="${MEMORY}" \
            --dependency="afterany:${training_job_id}" \
            --chdir="${repo_root}" \
            "${script_path}" --run-deferred "${repo_root}" "${run_dir_abs}"
    )"
    echo "  submitted eval job: ${eval_job_id}"
done

if [[ "${had_error}" == "true" ]]; then
    exit 1
fi
