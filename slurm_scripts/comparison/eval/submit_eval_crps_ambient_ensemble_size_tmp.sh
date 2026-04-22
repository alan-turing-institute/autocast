#!/bin/bash

set -euo pipefail
# Temporary eval submitter for the 2026-04-20 full-run CNS CRPS
# ensemble-size ablations. These are ambient/EPD checkpoints with
# model.n_members=16, kept separate from the main ambient eval script until
# the ablation set settles.
#
# Batch size: the standard ambient CRPS eval uses 8/GPU at n_members=10.
# These ablations raise n_members to 16, so 4/GPU is a conservative starting
# point for eval memory while preserving the same rollout/metric settings.

EVAL_BATCH_SIZE=4
TIMEOUT_MIN=240
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_5e157a5"  # ensemble_m16_fixed_bs32
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_dcd79e4"  # ensemble_m16_eff_bs1024
)

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting CRPS-ambient ensemble-size eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
