#!/bin/bash

set -euo pipefail

# Sequentially evaluate the eight main-comparison runs from SUMMARY_v1.md.
#
# Preview and preflight only (the default):
#   bash slurm_scripts/comparison/eval/\
#     run_main_comparison_trajectory_stats_20260801.sh
#
# Run with blocking interactive allocations, one at a time:
#   RUN=true bash slurm_scripts/comparison/eval/\
#     run_main_comparison_trajectory_stats_20260801.sh
# Resume from a named run after validating earlier outputs:
#   RUN=true START_AT=crps_cns bash slurm_scripts/comparison/eval/\
#     run_main_comparison_trajectory_stats_20260801.sh
#
# Every destination must be absent before the driver starts, except for the
# explicitly authorized empty CRPS AD directory below. New destinations are
# reserved with plain mkdir, and no non-empty destination is ever reused or
# removed by this script.

SUMMARY_PATH="/home/u6eo/ltcx7228.u6eo/autocast-02/outputs/2026-05-06_submission_outputs/SUMMARY_v1.md"
REUSABLE_EMPTY_OUTPUTS=(
    "/home/u6eo/ltcx7228.u6eo/autocast/outputs/2026-04-24/crps_ad64_vit_azula_large_bed4611_da01a04/eval_best_multiwinkler_from0p25_trajectory_stats_20260801"
    "/home/u6eo/ltcx7228.u6eo/autocast/outputs/2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534/eval_best_multiwinkler_from0p25_trajectory_stats_20260801"
)
EVAL_N_MEMBERS=10
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"
TEST_WINDOWS="[null]"
ROLLOUT_WINDOWS="[[0,1],[0,4],[6,12],[13,30],[31,99]]"

# label|run directory|checkpoint from SUMMARY_v1.md|autoencoder checkpoint|
# output subdirectory|eval mode|eval batch size|allocation time|memory
RUNS=(
    "crps_ad|outputs/2026-04-24/crps_ad64_vit_azula_large_bed4611_da01a04|autocast/h1pfa9dx/checkpoints/best-multiwinkler-from0p25-0477-0.0001.ckpt|null|eval_best_multiwinkler_from0p25_trajectory_stats_20260801|ambient|8|00:45:00|216G"
    "crps_cns|outputs/2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534|autocast/g3oxmc8x/checkpoints/best-multiwinkler-from0p25-0314-0.0063.ckpt|null|eval_best_multiwinkler_from0p25_trajectory_stats_20260801|ambient|8|00:45:00|216G"
    "crps_gpe|outputs/2026-04-24/crps_gpe64_vit_azula_large_bed4611_e0a6df5|autocast/9nqjv37c/checkpoints/best-multiwinkler-from0p25-0131-0.0040.ckpt|null|eval_best_multiwinkler_from0p25_trajectory_stats_20260801|ambient|8|00:45:00|216G"
    "crps_gs|outputs/2026-04-24/crps_gs64_vit_azula_large_bed4611_828a161|autocast/7p2z13p6/checkpoints/best-multiwinkler-from0p25-0382-0.0006.ckpt|null|eval_best_multiwinkler_from0p25_trajectory_stats_20260801|ambient|8|00:45:00|216G"
    "fm_ad|outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382|processor.ckpt|outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt|eval_trajectory_stats_20260801|encode_once|4|02:00:00|216G"
    "fm_cns|outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3|processor.ckpt|outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt|eval_trajectory_stats_20260801|encode_once|4|02:00:00|216G"
    "fm_gpe|outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a|processor.ckpt|outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt|eval_trajectory_stats_20260801|encode_once|4|02:00:00|216G"
    "fm_gs|outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331|processor.ckpt|outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt|eval_trajectory_stats_20260801|encode_once|4|02:00:00|216G"
)

run_eval() {
    if (( $# != 7 )); then
        echo "Usage: $0 --run-eval REPO_ROOT RUN_DIR CHECKPOINT AUTOENCODER_CHECKPOINT OUTPUT_SUBDIR MODE BATCH_SIZE" >&2
        return 2
    fi

    local repo_root="$1"
    local run_dir="$2"
    local checkpoint="$3"
    local autoencoder_checkpoint="$4"
    local output_subdir="$5"
    local eval_mode="$6"
    local eval_batch_size="$7"
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
        echo "Missing indexed checkpoint: ${checkpoint}" >&2
        return 1
    fi
    if [[ "${autoencoder_checkpoint}" != "null" && ! -f "${autoencoder_checkpoint}" ]]; then
        echo "Missing indexed autoencoder checkpoint: ${autoencoder_checkpoint}" >&2
        return 1
    fi
    if [[ -e "${trajectory_dir}" ]]; then
        echo "Refusing to reuse trajectory output path: ${trajectory_dir}" >&2
        return 1
    fi

    # Slurm opens these two new log files before the worker starts. Nothing
    # else may exist in the newly reserved destination at this point.
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

    cd "${repo_root}"
    source "${repo_root}/.venv/bin/activate"

    if [[ "${autoencoder_checkpoint}" == "null" ]]; then
        autoencoder_override="autoencoder_checkpoint=null"
    else
        autoencoder_override="+autoencoder_checkpoint=${autoencoder_checkpoint}"
    fi

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
        eval.compute_rollout_coverage=false \
        eval.compute_rollout_metrics=false \
        eval.batch_indices="[]" \
        eval.save_rollout_snapshots=false \
        eval.benchmark.enabled=false \
        eval.benchmark_rollout.enabled=false \
        eval.csv_path="${output_dir}/evaluation_metrics.csv" \
        eval.trajectory_statistics.enabled=true \
        eval.trajectory_statistics.output_dir="${trajectory_dir}" \
        eval.trajectory_statistics.overwrite_existing=false \
        eval.trajectory_statistics.sampling_seed=42 \
        eval.trajectory_statistics.include_per_timestep=true \
        logging.wandb.enabled=false
}

is_authorized_empty_output() {
    local output_dir="$1"
    local authorized_output
    local is_authorized=false
    local entries

    for authorized_output in "${REUSABLE_EMPTY_OUTPUTS[@]}"; do
        if [[ "${output_dir}" == "${authorized_output}" ]]; then
            is_authorized=true
            break
        fi
    done
    if [[ "${is_authorized}" != "true" ]]; then
        return 1
    fi
    if [[ ! -d "${output_dir}" || -L "${output_dir}" ]]; then
        return 1
    fi

    shopt -s nullglob dotglob
    entries=("${output_dir}"/*)
    shopt -u nullglob dotglob
    (( ${#entries[@]} == 0 ))
}

if [[ "${1:-}" == "--run-eval" ]]; then
    shift
    run_eval "$@"
    exit
fi

RUN="${RUN:-false}"
START_AT="${START_AT:-crps_ad}"
case "${RUN}" in
    true|false) ;;
    *)
        echo "RUN must be true or false, got: ${RUN}" >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
script_path="$(realpath "${BASH_SOURCE[0]}")"
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

if [[ ! -f "${SUMMARY_PATH}" ]]; then
    echo "Missing submission index: ${SUMMARY_PATH}" >&2
    exit 1
fi
if [[ ! -x "${repo_root}/.venv/bin/python" ]]; then
    echo "Missing project environment: ${repo_root}/.venv" >&2
    exit 1
fi

echo "Submission index: ${SUMMARY_PATH}"
echo "Repository: ${repo_root}"
echo "Execution enabled: ${RUN}"
echo "Starting at: ${START_AT}"

# Global read-only preflight: nothing is created unless every destination is
# absent and every source configuration/checkpoint is available.
for run_spec in "${SELECTED_RUNS[@]}"; do
    IFS='|' read -r label run_rel checkpoint_rel autoencoder_rel output_subdir \
        eval_mode eval_batch_size time_limit memory <<< "${run_spec}"
    run_dir="${repo_root}/${run_rel}"
    checkpoint="${run_dir}/${checkpoint_rel}"
    if [[ "${autoencoder_rel}" == "null" ]]; then
        autoencoder_checkpoint=null
    else
        autoencoder_checkpoint="${repo_root}/${autoencoder_rel}"
    fi
    output_dir="${run_dir}/${output_subdir}"

    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing run configuration: ${run_dir}/resolved_config.yaml" >&2
        exit 1
    fi
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing indexed checkpoint: ${checkpoint}" >&2
        exit 1
    fi
    if [[ "${autoencoder_checkpoint}" != "null" && ! -f "${autoencoder_checkpoint}" ]]; then
        echo "Missing indexed autoencoder checkpoint: ${autoencoder_checkpoint}" >&2
        exit 1
    fi
    if [[ -e "${output_dir}" ]]; then
        if ! is_authorized_empty_output "${output_dir}"; then
            echo "Refusing to reuse output path: ${output_dir}" >&2
            exit 1
        fi
        echo "  authorized empty-directory reuse: true"
    fi

    echo "plan: ${label}"
    echo "  checkpoint: ${checkpoint}"
    echo "  autoencoder_checkpoint: ${autoencoder_checkpoint}"
    echo "  output: ${output_dir}"
    echo "  mode: ${eval_mode}"
    echo "  time: ${time_limit}"
    echo "  memory: ${memory}"
done

if [[ "${RUN}" == "false" ]]; then
    echo "Preflight passed; set RUN=true to execute sequentially."
    exit
fi

for run_spec in "${SELECTED_RUNS[@]}"; do
    IFS='|' read -r label run_rel checkpoint_rel autoencoder_rel output_subdir \
        eval_mode eval_batch_size time_limit memory <<< "${run_spec}"
    run_dir="${repo_root}/${run_rel}"
    checkpoint="${run_dir}/${checkpoint_rel}"
    if [[ "${autoencoder_rel}" == "null" ]]; then
        autoencoder_checkpoint=null
    else
        autoencoder_checkpoint="${repo_root}/${autoencoder_rel}"
    fi
    output_dir="${run_dir}/${output_subdir}"

    # Recheck immediately before mutation. Only the one explicitly authorized
    # empty directory may already exist; all other paths are atomically reserved.
    if [[ -e "${output_dir}" ]]; then
        if ! is_authorized_empty_output "${output_dir}"; then
            echo "Refusing to reuse output path: ${output_dir}" >&2
            exit 1
        fi
    else
        mkdir "${output_dir}"
    fi

    echo "Starting interactive allocation for ${label}"
    if ! salloc \
        --job-name="traj_${label}" \
        --nodes=1 \
        --ntasks=1 \
        --gpus=1 \
        --time="${time_limit}" \
        --mem="${memory}" \
        srun \
            --nodes=1 \
            --ntasks=1 \
            --gpus=1 \
            --open-mode=append \
            --output="${output_dir}/slurm-%j.out" \
            --error="${output_dir}/slurm-%j.err" \
            bash "${script_path}" --run-eval \
                "${repo_root}" \
                "${run_dir}" \
                "${checkpoint}" \
                "${autoencoder_checkpoint}" \
                "${output_subdir}" \
                "${eval_mode}" \
                "${eval_batch_size}"; then
        echo "Evaluation failed for ${label}; preserving ${output_dir}" >&2
        exit 1
    fi

    trajectory_dir="${output_dir}/trajectory_statistics"
    required_outputs=(
        "${trajectory_dir}/single_step_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_trajectory.csv"
        "${trajectory_dir}/rollout_metrics_per_timestep_per_trajectory.csv"
    )
    for required_output in "${required_outputs[@]}"; do
        if [[ ! -s "${required_output}" ]]; then
            echo "Missing or empty required output: ${required_output}" >&2
            exit 1
        fi
    done
    echo "Validated trajectory statistics for ${label}"
done

echo "All eight trajectory-statistics evaluations completed successfully."
