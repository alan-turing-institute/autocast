#!/bin/bash

set -euo pipefail

# Evaluate planned updates batch 03 with the canonical ambient CRPS metrics.
#
# This batch contains the parameter-matched CNS CRPS FNO ablation.
# FNO_RUN_DIR may be supplied explicitly; otherwise use the latest matching
# production run under outputs/*/planned_updates_03/.

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

find_fno_run_dir() {
    if [[ ! -d outputs ]]; then
        return 0
    fi

    local checkpoint
    checkpoint="$(
        find outputs \
            -path "*/planned_updates_03/*/encoder_processor_decoder.ckpt" \
            | sort | tail -n 1
    )"
    if [[ -n "${checkpoint}" ]]; then
        dirname "${checkpoint}"
    fi
}

run_dir="${FNO_RUN_DIR:-}"
if [[ -z "${run_dir}" ]]; then
    run_dir="$(find_fno_run_dir)"
fi
if [[ -z "${run_dir}" ]] \
    || [[ ! -f "${run_dir}/resolved_config.yaml" ]] \
    || [[ ! -f "${run_dir}/encoder_processor_decoder.ckpt" ]]; then
    echo "FATAL: no completed FNO run found; set FNO_RUN_DIR explicitly" >&2
    exit 1
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
