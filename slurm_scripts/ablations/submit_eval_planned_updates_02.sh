#!/bin/bash

set -euo pipefail

# Evaluate the planned_updates_02 CNS MC-dropout run with 50 stochastic
# forward passes. The run directory is discovered from the W&B
# run name written into resolved_config.yaml by the production submitter.
#
# Preview:
#   RUN_ROOT=outputs/YYYY-MM-DD/planned_updates_02 \
#     ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh
#
# Submit:
#   RUN_ROOT=outputs/YYYY-MM-DD/planned_updates_02 SUBMIT=true \
#     ./slurm_scripts/ablations/submit_eval_planned_updates_02.sh

EVAL_BATCH_SIZE=1
EVAL_N_MEMBERS=50
TIMEOUT_MIN=180
EVAL_SUBDIR="eval_mc50_best_val"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"
RUN_ROOT="${RUN_ROOT:-outputs/$(date +%Y-%m-%d)/planned_updates_02}"
SUBMIT="${SUBMIT:-false}"

case "${SUBMIT}" in
    true|false) ;;
    *)
        echo "SUBMIT must be true or false, got: ${SUBMIT}" >&2
        exit 2
        ;;
esac

DATASETS=(
    # "gray_scott"
    # "gpe_laser_only_wake"
    "conditioned_navier_stokes"
    # "advection_diffusion"
)

had_error=false

for datamodule in "${DATASETS[@]}"; do
    run_id="mc_dropout_mse_${datamodule}"
    mapfile -t configs < <(
        find "${RUN_ROOT}" -type f -name resolved_config.yaml -print0 2>/dev/null \
            | xargs -0 grep -lF "name: ${run_id}" 2>/dev/null \
            | sort
    )

    if (( ${#configs[@]} != 1 )); then
        echo "Expected one resolved config for ${run_id}; found ${#configs[@]}" >&2
        had_error=true
        continue
    fi

    run_dir="$(dirname "${configs[0]}")"
    mapfile -t checkpoints < <(
        find "${run_dir}" -type f \
            -path '*/checkpoints/best-val-*.ckpt' -print | sort
    )
    if (( ${#checkpoints[@]} != 1 )); then
        echo "Expected one best-val checkpoint for ${run_id}; found ${#checkpoints[@]}" >&2
        had_error=true
        continue
    fi

    run_dir_abs="$(realpath "${run_dir}")"
    checkpoint_abs="$(realpath "${checkpoints[0]}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    echo "Submitting MC-dropout MSE + L2 evaluation"
    echo "  mode: $([[ "${SUBMIT}" == "true" ]] && echo slurm || echo preview)"
    echo "  run_id: ${run_id}"
    echo "  run_dir: ${run_dir_abs}"
    echo "  checkpoint: ${checkpoint_abs}"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"

    if [[ "${SUBMIT}" == "false" ]]; then
        continue
    fi

    uv run autocast eval --mode slurm \
        --workdir "${run_dir_abs}" \
        --output-subdir "${EVAL_SUBDIR}" \
        eval.checkpoint="${checkpoint_abs}" \
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
        eval.devices=1 \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}"
done

if [[ "${had_error}" == "true" ]]; then
    exit 1
fi
