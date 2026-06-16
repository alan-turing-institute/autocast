#!/bin/bash

set -euo pipefail
# Evaluate the RB effective-batch 24h CRPS-in-ambient EPD run
# (the_well_rayleigh_benard_effbatch24h_crps_ambient_b32_m8).
#
# This is an EPD checkpoint (encoder_processor_decoder.ckpt) with identity
# encoder/decoder and global_cond/AdaLN conditioning for constants/boundary
# conditions. Eval uses the resolved_config.yaml written alongside the run, so
# the trained architecture is reproduced exactly, and scores directly in raw
# ambient RB space -- there is no LoLA autoencoder in this path.
#
# Force eval.mode=ambient. Stateless identity EPD checkpoints can look
# processor-only to eval.mode=auto because the identity encoder/decoder add no
# encoder_decoder.* weights, but raw-space ambient rollout is the right route.
#
# Batch size: RB ambient resolution is 512x128 (16x the 64x64 Well-2D basis),
# so EVAL_BATCH_SIZE=1 even though CRPS is a single forward per rollout step (no
# ODE). No AE decode and no chunking are needed here (identity encoder/decoder).
#
# Single-GPU on purpose: matches the comparison/eval/ scripts (known-good
# wall-clock and outputs), avoids the DDP tail-padding bias for aggregate
# metrics, and keeps `_render_rollouts` correct -- under DDP the renderer races
# on shared video paths from each rank. For a faster split, follow up with a
# 4-GPU metrics-only job (`+distributed=ddp_4gpu_slurm eval.batch_indices=[]`)
# plus a 1-GPU render-only job rather than turning DDP on here.
#
# transpose_spatial=true only swaps the two spatial axes in rollout plots so the
# 512x128 RB fields render the right way up; it does not affect metrics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_MEMBERS="${EVAL_N_MEMBERS:-10}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
# LOLA RB paper figures use start=16 as the final conditioning timestep.
EVAL_ROLLOUT_START="${EVAL_ROLLOUT_START:-16}"
EVAL_MAX_ROLLOUT_STEPS="${EVAL_MAX_ROLLOUT_STEPS:-46}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_ambient_start${EVAL_ROLLOUT_START}}"
EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-true}"
EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-true}"
TIMEOUT_MIN="${TIMEOUT_MIN:-720}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_MEM="${SLURM_MEM:-256G}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-false}"
if [[ "${DRY_RUN_ONLY}" == "true" ]]; then
    RUN_DRY_STATES=("true")
else
    RUN_DRY_STATES=("true" "false")
fi
EVAL_METRICS="${EVAL_METRICS:-[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,vmse_v2,vrmse_v2,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,spread,skill,ssr,winkler]}"

RUN_PATTERNS=(
    "outputs/2026-05-20/the_well_rayleigh_benard_effbatch24h_crps_ambient_b32_m8"
)

RUN_DIRS=()
shopt -s nullglob
for run_pattern in "${RUN_PATTERNS[@]}"; do
    matches=( ${run_pattern} )
    if ((${#matches[@]} == 0)); then
        RUN_DIRS+=("${run_pattern}")
    else
        RUN_DIRS+=("${matches[@]}")
    fi
done
shopt -u nullglob

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi
    if [[ ! -f "${run_dir}/encoder_processor_decoder.ckpt" ]]; then
        echo "Skipping ${run_dir}: encoder_processor_decoder.ckpt missing" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting RB eff-batch 24h CRPS-ambient eval (mode=ambient, EPD)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.mode: ambient"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.rollout_start: ${EVAL_ROLLOUT_START}"
        echo "  eval.max_rollout_steps: ${EVAL_MAX_ROLLOUT_STEPS}"
        echo "  eval.transpose_spatial: true"
        echo "  eval.benchmark.enabled: ${EVAL_BENCHMARK_ENABLED}"
        echo "  eval.benchmark_rollout.enabled: ${EVAL_BENCHMARK_ROLLOUT_ENABLED}"
        echo "  hydra.launcher.mem: ${SLURM_MEM}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint=encoder_processor_decoder.ckpt \
            eval.mode=ambient \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            eval.rollout_start="${EVAL_ROLLOUT_START}" \
            eval.max_rollout_steps="${EVAL_MAX_ROLLOUT_STEPS}" \
            eval.transpose_spatial=true \
            eval.rollout_member_indices="${EVAL_DIAGNOSTIC_MEMBER_INDICES}" \
            eval.rollout_member_render_mode="${EVAL_ROLLOUT_MEMBER_RENDER_MODE}" \
            eval.benchmark.enabled="${EVAL_BENCHMARK_ENABLED}" \
            eval.benchmark_rollout.enabled="${EVAL_BENCHMARK_ROLLOUT_ENABLED}" \
            hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
            hydra.launcher.additional_parameters.mem="${SLURM_MEM}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
