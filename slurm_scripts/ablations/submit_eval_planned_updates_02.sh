#!/bin/bash

set -euo pipefail

# Submit the four planned_updates_02 MC-dropout MSE evaluations after their
# corresponding training jobs leave the queue.
#
# The final exported checkpoint is resolved inside the allocated evaluation
# job, after its afterany dependency is satisfied. This matches the checkpoint
# policy used for the non-multi-Winkler FM and diffusion evaluations.
#
# Preview all four submissions without queueing anything:
#   ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh
#
# Queue them:
#   SUBMIT=true ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh

EVAL_BATCH_SIZE=1
EVAL_N_MEMBERS=50
TIMEOUT_MIN=45
MEMORY="115G"
EVAL_SUBDIR="eval_mc50_final"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

run_deferred_eval() {
    local repo_root="$1"
    local run_dir="$2"

    cd "${repo_root}"

    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Missing resolved config: ${run_dir}/resolved_config.yaml" >&2
        return 1
    fi

    local eval_ckpt="${run_dir}/encoder_processor_decoder.ckpt"
    if [[ ! -f "${eval_ckpt}" ]]; then
        echo "Missing final checkpoint: ${eval_ckpt}" >&2
        return 1
    fi

    local eval_ckpt_abs eval_output_dir
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir}/${EVAL_SUBDIR}"

    echo "Starting deferred MC-dropout MSE ambient evaluation"
    echo "  run_dir: ${run_dir}"
    echo "  eval.checkpoint: ${eval_ckpt_abs}"
    echo "  eval.mode: ambient"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
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

RUN_ROOT="${RUN_ROOT:-outputs/2026-07-27/planned_updates_02}"
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
    "mcdo_mse_ad|5809381|epd_ad64_vit_azula_mc_dropout_large_5d39424_282ad08"
    "mcdo_mse_gs|5809382|epd_gs64_vit_azula_mc_dropout_large_5d39424_f731d87"
    "mcdo_mse_gpe|5809385|epd_gpe64_vit_azula_mc_dropout_large_5d39424_234111d"
    "mcdo_mse_cns|5809390|epd_cns64_vit_azula_mc_dropout_large_5d39424_4c7c6be"
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
    echo "  deferred checkpoint: encoder_processor_decoder.ckpt"
    echo "  eval.mode: ambient"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  output_subdir: ${EVAL_SUBDIR}"
    echo "  time: ${TIMEOUT_MIN} minutes"
    echo "  memory: ${MEMORY}"

    if [[ "${SUBMIT}" == "false" ]]; then
        continue
    fi

    mkdir -p "${eval_output_dir}"
    eval_job_id="$(
        sbatch --parsable \
            --job-name="eval_${run_id}_final" \
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
