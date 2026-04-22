#!/bin/bash

set -euo pipefail
# Evaluate FM-in-ambient (flow matching, identity encoder) EPD runs from
# 2026-04-18 across all 4 datasets. Eval reuses resolved_config.yaml so
# flow_ode_steps (=50), hid_channels, and backbone match training.
#
# Batch size: diffusion rollout is ODE-integrated (flow_ode_steps=50) per
# rollout step, so ambient 64x64 × n_members=10 × 50 ODE substeps is the
# tightest of the three. 4/GPU fits; drop to 2 if OOM.
#
# We also pin eval.n_members explicitly here so the comparison scripts do not
# depend on the global eval default staying at 10.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-18/diff_gs64_flow_matching_vit_0f89f06_6e3a299"
    "outputs/2026-04-18/diff_gpe64_flow_matching_vit_0f89f06_3b3604d"
    "outputs/2026-04-18/diff_cns64_flow_matching_vit_0f89f06_483bb70"
    "outputs/2026-04-18/diff_ad64_flow_matching_vit_0f89f06_725d44a"
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

        echo "Submitting FM-ambient eval"
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
