#!/bin/bash

set -euo pipefail

# Sequentially evaluate the seed-43 CNS main-comparison reruns using the
# trajectory-statistics implementation and environment from ~/autocast.
#
# Preview and preflight only (default; does not create output directories):
#   bash slurm_scripts/comparison/eval/\
#     run_seed43_main_comparison_eval_20260804.sh
#
# Run both evaluations sequentially through the interactive reservation:
#   RUN=true bash slurm_scripts/comparison/eval/\
#     run_seed43_main_comparison_eval_20260804.sh
# Resume at FM after a separately validated CRPS evaluation:
#   RUN=true START_AT=fm_cns_seed43 bash slurm_scripts/comparison/eval/\
#     run_seed43_main_comparison_eval_20260804.sh
#
# Every destination must be absent before the driver starts. The driver
# reserves each destination with plain mkdir immediately before its blocking
# srun. It never removes or reuses an evaluation directory.

EXEC_REPO="/home/u6eo/ltcx7228.u6eo/autocast"
RUN_ROOT="/home/u6eo/ltcx7228.u6eo/autocast-02/outputs/2026-08-01/main_comparison_cns_seed43"
PUBLISHED_AE_CHECKPOINT="/home/u6eo/ltcx7228.u6eo/autocast-02/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
TRAJECTORY_STATS_COMMIT="e452908b"

RESERVATION="${RESERVATION:-interactive}"
MEMORY="115G"
EVAL_N_MEMBERS=10
VISUAL_BATCH_INDICES="[0,1,2,3,4,5,6,7]"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
TEST_WINDOWS="[null]"
ROLLOUT_WINDOWS="[[0,1],[0,4],[6,12],[13,30],[31,99]]"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,spread,skill,ssr,winkler]"

CRPS_RUN_DIR="${RUN_ROOT}/crps_vit_azula_large"
CRPS_CHECKPOINT="${CRPS_RUN_DIR}/autocast/spy78y66/checkpoints/best-multiwinkler-overall-0255-0.0066.ckpt"
CRPS_OUTPUT_SUBDIR="eval_best_multiwinkler_overall_trajectory_stats_20260804"

FM_RUN_DIR="${RUN_ROOT}/fm_vit_large_published_ae"
FM_CHECKPOINT="${FM_RUN_DIR}/processor.ckpt"
FM_OUTPUT_SUBDIR="eval_trajectory_stats_20260804_retry1"

# label|run directory|checkpoint|AE checkpoint or null|output subdirectory|
# eval mode|eval batch size|allocation time
RUNS=(
    "crps_cns_seed43|${CRPS_RUN_DIR}|${CRPS_CHECKPOINT}|null|${CRPS_OUTPUT_SUBDIR}|ambient|8|00:45:00"
    "fm_cns_seed43|${FM_RUN_DIR}|${FM_CHECKPOINT}|${PUBLISHED_AE_CHECKPOINT}|${FM_OUTPUT_SUBDIR}|encode_once|4|02:00:00"
)

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

run_eval() {
    if (( $# != 7 )); then
        echo "Usage: $0 --run-eval RUN_DIR CHECKPOINT AE_CHECKPOINT OUTPUT_SUBDIR MODE BATCH_SIZE EXEC_REPO" >&2
        return 2
    fi

    local run_dir="$1"
    local checkpoint="$2"
    local autoencoder_checkpoint="$3"
    local output_subdir="$4"
    local eval_mode="$5"
    local eval_batch_size="$6"
    local exec_repo="$7"
    local output_dir="${run_dir}/${output_subdir}"
    local trajectory_dir="${output_dir}/trajectory_statistics"
    local autoencoder_override
    local entry basename

    if [[ ! -d "${output_dir}" || -L "${output_dir}" ]]; then
        echo "Missing newly reserved output directory: ${output_dir}" >&2
        return 1
    fi
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing run configuration: ${run_dir}/resolved_config.yaml" >&2
        return 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing evaluation checkpoint: ${checkpoint}" >&2
        return 1
    fi
    if [[ "${autoencoder_checkpoint}" != "null" && ! -f "${autoencoder_checkpoint}" ]]; then
        echo "Missing autoencoder checkpoint: ${autoencoder_checkpoint}" >&2
        return 1
    fi
    if [[ -e "${trajectory_dir}" || -L "${trajectory_dir}" ]]; then
        echo "Refusing to reuse trajectory output path: ${trajectory_dir}" >&2
        return 1
    fi

    # Slurm may create these log files before the worker starts. Nothing else
    # is permitted in the newly reserved output directory.
    shopt -s nullglob dotglob
    for entry in "${output_dir}"/*; do
        basename="${entry##*/}"
        case "${basename}" in
            slurm-*.out|slurm-*.err) ;;
            *)
                echo "Unexpected pre-existing output entry: ${entry}" >&2
                return 1
                ;;
        esac
    done
    shopt -u nullglob dotglob

    if [[ "${autoencoder_checkpoint}" == "null" ]]; then
        autoencoder_override="autoencoder_checkpoint=null"
    else
        autoencoder_override="+autoencoder_checkpoint=${autoencoder_checkpoint}"
    fi

    cd "${exec_repo}"
    source "${exec_repo}/.venv/bin/activate"

    exec uv run --active --no-sync autocast eval --mode local \
        --workdir "${run_dir}" \
        --output-subdir "${output_subdir}" \
        eval.checkpoint="${checkpoint}" \
        "${autoencoder_override}" \
        eval.mode="${eval_mode}" \
        eval.accelerator=cuda \
        eval.devices=1 \
        eval.batch_size="${eval_batch_size}" \
        eval.n_members="${EVAL_N_MEMBERS}" \
        eval.max_test_batches=null \
        eval.max_rollout_batches=null \
        eval.max_rollout_steps=25 \
        eval.metric_windows="${TEST_WINDOWS}" \
        eval.metric_windows_rollout="${ROLLOUT_WINDOWS}" \
        eval.metrics="${EVAL_METRICS}" \
        eval.compute_rollout_coverage=true \
        eval.compute_rollout_metrics=true \
        eval.batch_indices="${VISUAL_BATCH_INDICES}" \
        eval.save_rollout_snapshots=true \
        eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
        eval.rollout_snapshot_format=png \
        eval.benchmark.enabled=true \
        eval.benchmark_rollout.enabled=true \
        eval.csv_path="${output_dir}/evaluation_metrics.csv" \
        eval.video_dir="${output_dir}/videos" \
        eval.trajectory_statistics.enabled=true \
        eval.trajectory_statistics.output_dir="${trajectory_dir}" \
        eval.trajectory_statistics.overwrite_existing=false \
        eval.trajectory_statistics.sampling_seed=42 \
        eval.trajectory_statistics.include_per_timestep=true \
        logging.wandb.enabled=false
}

if [[ "${1:-}" == "--run-eval" ]]; then
    shift
    run_eval "$@"
    exit
fi

RUN="${RUN:-false}"
START_AT="${START_AT:-crps_cns_seed43}"
case "${RUN}" in
    true|false) ;;
    *)
        echo "RUN must be true or false, got: ${RUN}" >&2
        exit 2
        ;;
esac

start_found=false
SELECTED_RUNS=()
for run_spec in "${RUNS[@]}"; do
    IFS='|' read -r label _ <<< "${run_spec}"
    if [[ "${label}" == "${START_AT}" ]]; then
        start_found=true
    fi
    if [[ "${start_found}" == "true" ]]; then
        SELECTED_RUNS+=("${run_spec}")
    fi
done
if [[ "${start_found}" != "true" ]]; then
    echo "Unknown START_AT label: ${START_AT}" >&2
    exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
script_path="$(realpath "${BASH_SOURCE[0]}")"

if [[ "${repo_root}" != "/home/u6eo/ltcx7228.u6eo/autocast-02" ]]; then
    fail "Run this script from the autocast-02 worktree; resolved ${repo_root}"
fi
if [[ ! -d "${EXEC_REPO}/.git" ]]; then
    fail "Missing execution repository: ${EXEC_REPO}"
fi
if [[ ! -x "${EXEC_REPO}/.venv/bin/python" ]]; then
    fail "Missing execution environment: ${EXEC_REPO}/.venv"
fi
if ! git -C "${EXEC_REPO}" merge-base --is-ancestor \
    "${TRAJECTORY_STATS_COMMIT}" HEAD; then
    fail "${EXEC_REPO} HEAD does not contain trajectory-statistics commit ${TRAJECTORY_STATS_COMMIT}"
fi

# Strictly confirm that the pinned CRPS checkpoint is the one and only
# overall-best multi-Winkler checkpoint produced by this completed run.
shopt -s nullglob
crps_overall_checkpoints=(
    "${CRPS_RUN_DIR}"/autocast/*/checkpoints/best-multiwinkler-overall-*.ckpt
)
shopt -u nullglob
if (( ${#crps_overall_checkpoints[@]} != 1 )); then
    fail "Expected exactly one overall-best multi-Winkler checkpoint; found ${#crps_overall_checkpoints[@]}"
fi
if [[ "$(realpath "${crps_overall_checkpoints[0]}")" != "$(realpath "${CRPS_CHECKPOINT}")" ]]; then
    fail "Pinned CRPS checkpoint does not match the unique overall-best checkpoint"
fi

echo "Execution repository: ${EXEC_REPO}"
echo "Execution environment: ${EXEC_REPO}/.venv"
echo "Reservation: ${RESERVATION}"
echo "Memory per evaluation: ${MEMORY}"
echo "Execution enabled: ${RUN}"
echo "Starting at: ${START_AT}"

# Global read-only preflight: no directory is created unless every source is
# present and both destinations are absent.
for run_spec in "${SELECTED_RUNS[@]}"; do
    IFS='|' read -r label run_dir checkpoint autoencoder_checkpoint \
        output_subdir eval_mode eval_batch_size time_limit <<< "${run_spec}"
    output_dir="${run_dir}/${output_subdir}"

    if [[ ! -d "${run_dir}" || -L "${run_dir}" ]]; then
        fail "Missing or unsafe run directory for ${label}: ${run_dir}"
    fi
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        fail "Missing resolved config for ${label}: ${run_dir}/resolved_config.yaml"
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        fail "Missing checkpoint for ${label}: ${checkpoint}"
    fi
    if [[ "${autoencoder_checkpoint}" != "null" && ! -f "${autoencoder_checkpoint}" ]]; then
        fail "Missing autoencoder checkpoint for ${label}: ${autoencoder_checkpoint}"
    fi
    if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
        fail "Refusing to reuse output path for ${label}: ${output_dir}"
    fi

    echo "plan: ${label}"
    echo "  run_dir: ${run_dir}"
    echo "  checkpoint: ${checkpoint}"
    echo "  autoencoder_checkpoint: ${autoencoder_checkpoint}"
    echo "  eval.mode: ${eval_mode}"
    echo "  eval.batch_size: ${eval_batch_size}"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  output: ${output_dir}"
    echo "  time: ${time_limit}"
    echo "  memory: ${MEMORY}"
done

if [[ "${RUN}" == "false" ]]; then
    echo "Preflight passed; set RUN=true to execute the selected evaluations sequentially."
    exit
fi

for run_spec in "${SELECTED_RUNS[@]}"; do
    IFS='|' read -r label run_dir checkpoint autoencoder_checkpoint \
        output_subdir eval_mode eval_batch_size time_limit <<< "${run_spec}"
    output_dir="${run_dir}/${output_subdir}"
    trajectory_dir="${output_dir}/trajectory_statistics"

    if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
        fail "Refusing to reuse output path immediately before ${label}: ${output_dir}"
    fi
    mkdir "${output_dir}"

    echo "Starting blocking interactive-reservation srun for ${label}"
    if ! srun \
        --reservation="${RESERVATION}" \
        --job-name="eval_${label}" \
        --nodes=1 \
        --ntasks=1 \
        --gpus=1 \
        --time="${time_limit}" \
        --mem="${MEMORY}" \
        --open-mode=append \
        --output="${output_dir}/slurm-%j.out" \
        --error="${output_dir}/slurm-%j.err" \
        bash "${script_path}" --run-eval \
            "${run_dir}" \
            "${checkpoint}" \
            "${autoencoder_checkpoint}" \
            "${output_subdir}" \
            "${eval_mode}" \
            "${eval_batch_size}" \
            "${EXEC_REPO}"; then
        fail "Evaluation failed for ${label}; preserving ${output_dir} for diagnosis"
    fi

    required_outputs=(
        "${trajectory_dir}/resolved_eval_config.yaml"
        "${trajectory_dir}/evaluation_metrics.csv"
        "${trajectory_dir}/evaluation_metadata.csv"
        "${trajectory_dir}/benchmark_metrics.csv"
        "${trajectory_dir}/rollout_metrics.csv"
        "${trajectory_dir}/single_step_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_timestep_per_trajectory.csv"
    )
    for required_output in "${required_outputs[@]}"; do
        if [[ ! -s "${required_output}" ]]; then
            fail "Missing or empty required output for ${label}: ${required_output}"
        fi
    done

    shopt -s nullglob
    rollout_videos=("${trajectory_dir}"/videos/*.mp4)
    rollout_snapshots=("${trajectory_dir}"/videos/snapshots/*.png)
    shopt -u nullglob
    if (( ${#rollout_videos[@]} < 8 )); then
        fail "Expected at least 8 rollout videos for ${label}; found ${#rollout_videos[@]}"
    fi
    if (( ${#rollout_snapshots[@]} < 24 )); then
        fail "Expected at least 24 rollout snapshots for ${label}; found ${#rollout_snapshots[@]}"
    fi

    echo "Validated complete main-comparison evaluation for ${label}"
done

echo "Both seed-43 main-comparison evaluations completed successfully."
