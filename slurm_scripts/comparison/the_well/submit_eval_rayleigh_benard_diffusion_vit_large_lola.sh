#!/bin/bash

set -euo pipefail
# Evaluate Rayleigh-Benard LoLA diffusion ViT-large cached-latent runs using
# eval.mode=encode_once.
#
# These runs are processor-only diffusion ViTs trained on cached LoLA latents.
# There is no Lightning autoencoder.ckpt: the LoLA autoencoder is discovered
# automatically by the eval pipeline by walking up from datamodule.data_path
# until it finds config.yaml + state.pth. The wrapped encoder/decoder carry
# their own mean/std internally, and the eval pipeline substitutes the cached
# latents datamodule for an unnormalized raw TheWellDataModule.
#
# Single-GPU on purpose: matches the comparison/eval/ scripts, avoids DDP
# tail-padding bias for aggregate metrics, and keeps rollout rendering safe on
# shared output paths. For a faster split, use a 4-GPU metrics-only job plus a
# 1-GPU render-only job rather than enabling DDP here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_MEMBERS="${EVAL_N_MEMBERS:-10}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-8}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
# LOLA RB paper figures use start=16 as the final conditioning timestep.
EVAL_ROLLOUT_START="${EVAL_ROLLOUT_START:-16}"
EVAL_MAX_ROLLOUT_STEPS="${EVAL_MAX_ROLLOUT_STEPS:-46}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_encode_once_start${EVAL_ROLLOUT_START}}"
EVAL_AUTOENCODED_TARGET_METRICS="${EVAL_AUTOENCODED_TARGET_METRICS:-true}"
EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-true}"
EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-true}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
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
    "outputs/2026-05-22/the_well_rayleigh_benard_diffusion_vit_large_lola4096"
)

RUN_DIRS=()
shopt -s nullglob
for run_pattern in "${RUN_PATTERNS[@]}"; do
    matches=( ${run_pattern} )
    for match in "${matches[@]}"; do
        RUN_DIRS+=("${match}")
    done
done
shopt -u nullglob

if ((${#RUN_DIRS[@]} == 0)); then
    echo "No matching Rayleigh-Benard LoLA diffusion runs found." >&2
    exit 1
fi

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi
    if [[ ! -e "${run_dir}/processor.ckpt" ]]; then
        echo "Skipping ${run_dir}: processor.ckpt missing" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting RB LoLA diffusion ViT-large cached-latent eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.mode: encode_once"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.chunk_size: ${EVAL_CHUNK_SIZE}"
        echo "  eval.rollout_start: ${EVAL_ROLLOUT_START}"
        echo "  eval.max_rollout_steps: ${EVAL_MAX_ROLLOUT_STEPS}"
        echo "  eval.transpose_spatial: true"
        echo "  eval.compute_rollout_autoencoded_target_metrics: ${EVAL_AUTOENCODED_TARGET_METRICS}"
        echo "  eval.benchmark.enabled: ${EVAL_BENCHMARK_ENABLED}"
        echo "  eval.benchmark_rollout.enabled: ${EVAL_BENCHMARK_ROLLOUT_ENABLED}"
        echo "  hydra.launcher.mem: ${SLURM_MEM}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint=processor.ckpt \
            eval.mode=encode_once \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            +eval.chunk_size="${EVAL_CHUNK_SIZE}" \
            eval.rollout_start="${EVAL_ROLLOUT_START}" \
            eval.max_rollout_steps="${EVAL_MAX_ROLLOUT_STEPS}" \
            eval.transpose_spatial=true \
            eval.compute_rollout_autoencoded_target_metrics="${EVAL_AUTOENCODED_TARGET_METRICS}" \
            eval.rollout_member_indices="${EVAL_DIAGNOSTIC_MEMBER_INDICES}" \
            eval.rollout_member_render_mode="${EVAL_ROLLOUT_MEMBER_RENDER_MODE}" \
            eval.benchmark.enabled="${EVAL_BENCHMARK_ENABLED}" \
            eval.benchmark_rollout.enabled="${EVAL_BENCHMARK_ROLLOUT_ENABLED}" \
            hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
            hydra.launcher.additional_parameters.mem="${SLURM_MEM}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
