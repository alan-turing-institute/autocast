#!/bin/bash

set -euo pipefail

# Evaluate the two selected Gray-Scott runs while rendering every one of the
# 24 test trajectories. Output directories are dated and reserved atomically;
# an existing path is always treated as an error.
#
# Preview without creating directories or submitting jobs:
#   ./slurm_scripts/comparison/eval/submit_eval_gs_all_test_rollouts_20260801.sh
#
# Submit both jobs:
#   SUBMIT=true ./slurm_scripts/comparison/eval/submit_eval_gs_all_test_rollouts_20260801.sh

ROLLOUT_INDICES="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
EVAL_N_MEMBERS=10
TIMEOUT_MIN=60
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

run_eval() {
    local repo_root="$1"
    local run_dir="$2"
    local checkpoint="$3"
    local eval_subdir="$4"
    local eval_batch_size="$5"
    local eval_output_dir="${run_dir}/${eval_subdir}"
    local artifact

    cd "${repo_root}"

    if [[ ! -d "${eval_output_dir}" ]]; then
        echo "Missing newly reserved output directory: ${eval_output_dir}" >&2
        return 1
    fi

    # Slurm creates its log files before this script starts. Refuse to run if
    # the directory contains any evaluator artifact from an earlier attempt.
    for artifact in \
        resolved_eval_config.yaml \
        evaluation_metrics.csv \
        evaluation_metadata.csv \
        benchmark_metrics.csv \
        rollout_metrics.csv \
        videos; do
        if [[ -e "${eval_output_dir}/${artifact}" ]]; then
            echo "Refusing to overwrite existing eval artifact: ${eval_output_dir}/${artifact}" >&2
            return 1
        fi
    done

    srun --nodes=1 --ntasks=1 --gpus=1 \
        uv run autocast eval --mode local \
            --workdir "${run_dir}" \
            --output-subdir "${eval_subdir}" \
            eval.checkpoint="${checkpoint}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.batch_indices="${ROLLOUT_INDICES}" \
            eval.video_sample_index=0 \
            eval.fps=5 \
            eval.preserve_aspect=true \
            eval.save_rollout_snapshots=true \
            eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
            eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
            eval.rollout_snapshot_channels=null \
            eval.rollout_snapshot_format=png \
            eval.max_rollout_steps=25 \
            eval.max_test_batches=null \
            eval.max_rollout_batches=null \
            eval.compute_rollout_coverage=true \
            eval.compute_rollout_metrics=true \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${eval_batch_size}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            eval.devices=1
}

if [[ "${1:-}" == "--run-eval" ]]; then
    if (( $# != 6 )); then
        echo "Usage: $0 --run-eval REPO_ROOT RUN_DIR CHECKPOINT EVAL_SUBDIR EVAL_BATCH_SIZE" >&2
        exit 2
    fi
    run_eval "$2" "$3" "$4" "$5" "$6"
    exit
fi

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

# label|run directory|checkpoint relative to run|output subdir|batch size|memory
RUNS=(
    "vit_gs|outputs/2026-04-24/crps_gs64_vit_azula_large_bed4611_828a161|autocast/7p2z13p6/checkpoints/best-multiwinkler-from0p25-0382-0.0006.ckpt|eval_best_multiwinkler_from0p25_all24_test_rollouts_20260801|8|216G"
    "fno_gs|outputs/2026-07-27/planned_updates_03/crps_default_fno_concat_6c8145f_8940bbc|autocast/zlqe4wir/checkpoints/best-multiwinkler-overall-0584-0.0029.ckpt|eval_best_multiwinkler_overall_all24_test_rollouts_20260801|4|115G"
)

for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r label run_dir checkpoint_rel eval_subdir eval_batch_size memory <<< "${run_spec}"
    run_dir_abs="$(realpath "${run_dir}")"
    checkpoint_abs="${run_dir_abs}/${checkpoint_rel}"
    eval_output_dir="${run_dir_abs}/${eval_subdir}"

    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Missing run config: ${run_dir_abs}/resolved_config.yaml" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint_abs}" ]]; then
        echo "Missing checkpoint: ${checkpoint_abs}" >&2
        exit 1
    fi
    if [[ -e "${eval_output_dir}" ]]; then
        echo "Refusing to reuse existing output path: ${eval_output_dir}" >&2
        exit 1
    fi

    echo "plan: ${label}"
    echo "  run_dir: ${run_dir_abs}"
    echo "  checkpoint: ${checkpoint_abs}"
    echo "  output_dir: ${eval_output_dir}"
    echo "  rollout_indices: ${ROLLOUT_INDICES}"
    echo "  eval.batch_size: ${eval_batch_size}"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  time: ${TIMEOUT_MIN} minutes"
    echo "  memory: ${memory}"

    if [[ "${SUBMIT}" == "false" ]]; then
        continue
    fi

    # Atomic reservation: mkdir (without -p) fails if anything appeared at the
    # target path after the preflight check.
    mkdir "${eval_output_dir}"
    job_id="$(
        sbatch --parsable \
            --job-name="eval_${label}_all24" \
            --output="${eval_output_dir}/slurm-%j.out" \
            --error="${eval_output_dir}/slurm-%j.err" \
            --time="${TIMEOUT_MIN}" \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --mem="${memory}" \
            --chdir="${repo_root}" \
            "${script_path}" --run-eval "${repo_root}" "${run_dir_abs}" \
            "${checkpoint_abs}" "${eval_subdir}" "${eval_batch_size}"
    )"
    echo "  submitted job: ${job_id}"
done
