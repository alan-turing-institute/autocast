#!/bin/bash

set -euo pipefail

# Evaluate planned updates batch 03 with the canonical ambient CRPS metrics.
#
# This batch contains the four parameter-matched CRPS FNO runs.
# FNO_RUN_DIR may select one run explicitly; otherwise evaluate all completed
# runs in RUN_ROOT or the latest outputs/*/planned_updates_03/ batch.

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

find_latest_run_root() {
    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -type d -path "*/planned_updates_03" | sort | tail -n 1
}

RUN_DIRS=()
if [[ -n "${FNO_RUN_DIR:-}" ]]; then
    RUN_DIRS=("${FNO_RUN_DIR}")
else
    run_root="${RUN_ROOT:-}"
    if [[ -z "${run_root}" ]]; then
        run_root="$(find_latest_run_root)"
    fi
    if [[ -n "${run_root}" ]]; then
        mapfile -t RUN_DIRS < <(
            find "${run_root}" -type f -name encoder_processor_decoder.ckpt \
                | sort | while read -r checkpoint; do dirname "${checkpoint}"; done
        )
    fi
fi
if (( ${#RUN_DIRS[@]} == 0 )); then
    echo "FATAL: no completed FNO runs found; set RUN_ROOT or FNO_RUN_DIR" >&2
    exit 1
fi

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]] \
        || [[ ! -f "${run_dir}/encoder_processor_decoder.ckpt" ]]; then
        echo "Skipping incomplete FNO run: ${run_dir}" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting planned updates batch 03 evaluation"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.mode=ambient \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
