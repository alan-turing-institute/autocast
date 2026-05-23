#!/bin/bash

set -euo pipefail
# Evaluate the RB effective-batch 24h CRPS-in-ambient LoLA-pixel ViT run
# (the_well_rayleigh_benard_effbatch_crps_ambient_lola_pixel_b32_m8_24hr).
#
# This is an EPD checkpoint (encoder_processor_decoder.ckpt) that scores
# directly in raw ambient RB space. The processor uses the LoLA pixel-space ViT
# hyperparameters, but there is no LoLA autoencoder in this eval path.
#
# Force eval.mode=ambient. Stateless identity EPD checkpoints can look
# processor-only to eval.mode=auto because the identity encoder/decoder add no
# encoder_decoder.* weights, but raw-space ambient rollout is the right route.
#
# Single-GPU on purpose: matches the comparison/eval/ scripts, avoids DDP
# tail-padding bias for aggregate metrics, and keeps rollout rendering safe on
# shared output paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
cd "${REPO_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_MEMBERS="${EVAL_N_MEMBERS:-10}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-true}"
EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-true}"
TIMEOUT_MIN="${TIMEOUT_MIN:-720}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_MEM="${SLURM_MEM:-115G}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-false}"
if [[ "${DRY_RUN_ONLY}" == "true" ]]; then
    RUN_DRY_STATES=("true")
else
    RUN_DRY_STATES=("true" "false")
fi
EVAL_METRICS="${EVAL_METRICS:-[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,vmse_v2,vrmse_v2,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]}"

RUN_DIR="outputs/2026-05-22/the_well_rayleigh_benard_effbatch_crps_ambient_lola_pixel_b32_m8_24hr"

if [[ ! -f "${RUN_DIR}/resolved_config.yaml" ]]; then
    echo "Skipping ${RUN_DIR}: resolved_config.yaml missing" >&2
    exit 1
fi
if [[ ! -f "${RUN_DIR}/encoder_processor_decoder.ckpt" ]]; then
    echo "Skipping ${RUN_DIR}: encoder_processor_decoder.ckpt missing" >&2
    exit 1
fi

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting RB eff-batch 24h CRPS ambient LoLA-pixel eval"
    echo "  mode: ${run_label}"
    echo "  run_dir: ${RUN_DIR}"
    echo "  eval.mode: ambient"
    echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  eval.transpose_spatial: true"
    echo "  eval.benchmark.enabled: ${EVAL_BENCHMARK_ENABLED}"
    echo "  eval.benchmark_rollout.enabled: ${EVAL_BENCHMARK_ROLLOUT_ENABLED}"
    echo "  hydra.launcher.mem: ${SLURM_MEM}"
    echo "  eval.metrics: ${EVAL_METRICS}"

    uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
        --workdir "${RUN_DIR}" \
        eval.checkpoint=encoder_processor_decoder.ckpt \
        eval.mode=ambient \
        eval.metrics="${EVAL_METRICS}" \
        eval.batch_size="${EVAL_BATCH_SIZE}" \
        eval.n_members="${EVAL_N_MEMBERS}" \
        eval.transpose_spatial=true \
        eval.rollout_member_indices="${EVAL_DIAGNOSTIC_MEMBER_INDICES}" \
        eval.rollout_member_render_mode="${EVAL_ROLLOUT_MEMBER_RENDER_MODE}" \
        eval.benchmark.enabled="${EVAL_BENCHMARK_ENABLED}" \
        eval.benchmark_rollout.enabled="${EVAL_BENCHMARK_ROLLOUT_ENABLED}" \
        hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
        hydra.launcher.additional_parameters.mem="${SLURM_MEM}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}"
done
