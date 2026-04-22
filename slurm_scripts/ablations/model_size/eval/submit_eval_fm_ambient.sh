#!/bin/bash

set -euo pipefail
# Ambient eval submitter for the current FM model-size ablation runs.
#
# This stays under the ablation directory while the sweep remains CNS-only and
# only the larger preliminary leg is in hand. If the resulting run set becomes
# part of the canonical comparison study later, move the promoted run dirs into
# slurm_scripts/comparison/eval/ and leave ablation-only evals local here.
#
# We keep eval.n_members fixed at 10 to match the comparison-study eval regime.
#
# Batch size: baseline ambient FM eval fits 4/GPU at hid_channels=704,
# hid_blocks=12, and flow_ode_steps=50. The current 2x run is substantially
# larger (896/16), so start at 2/GPU for the preliminary pass and increase only
# after confirming cluster headroom.

EVAL_BATCH_SIZE=2
EVAL_N_MEMBERS=10
TIMEOUT_MIN=480
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    # Uncomment/add once the 0p4x production run exists.
    # "outputs/2026-04-21/model_size/diff_cns64_flow_matching_vit_472_<git>_<uuid>"  # model_size_fm_0p4x
    "outputs/2026-04-21/model_size/diff_cns64_flow_matching_vit_896_3a69487_e894c55"  # model_size_fm_2x
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

        echo "Submitting model-size FM ambient eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
