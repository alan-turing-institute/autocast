#!/bin/bash

set -euo pipefail
# Metrics-only recovery eval for the Rayleigh-Benard LoLA diffusion ViT-large
# cached-latent run.
#
# The full eval renders rollout videos before computing rollout-window metrics.
# If rendering dies partway through, the per-lead rollout coverage plots are
# never written. This script disables rendering/snapshots and benchmarks so the
# job goes straight to test + rollout metrics.
#
# By default this inherits the sampler from the run config (Euler-50 for the
# baseline run). Set EVAL_SAMPLER/EVAL_SAMPLER_STEPS/EVAL_SAMPLER_ORDER to reuse
# the same metrics-only path for sampler variants such as AB16 or DDPM50.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_N_MEMBERS="${EVAL_N_MEMBERS:-10}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-8}"
EVAL_SAMPLER="${EVAL_SAMPLER:-}"
EVAL_SAMPLER_STEPS="${EVAL_SAMPLER_STEPS:-}"
EVAL_SAMPLER_ORDER="${EVAL_SAMPLER_ORDER:-}"
# LOLA RB paper figures use start=16 as the final conditioning timestep.
EVAL_ROLLOUT_START="${EVAL_ROLLOUT_START:-16}"
if [[ -z "${EVAL_SUBDIR:-}" ]]; then
    if [[ -n "${EVAL_SAMPLER}" ]]; then
        sampler_suffix="${EVAL_SAMPLER}${EVAL_SAMPLER_STEPS}"
        if [[ -n "${EVAL_SAMPLER_ORDER}" ]]; then
            sampler_suffix="${sampler_suffix}_o${EVAL_SAMPLER_ORDER}"
        fi
        EVAL_SUBDIR="eval_encode_once_${sampler_suffix}_metrics_only_start${EVAL_ROLLOUT_START}"
    else
        EVAL_SUBDIR="eval_encode_once_metrics_only_start${EVAL_ROLLOUT_START}"
    fi
fi
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_MEM="${SLURM_MEM:-115G}"
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

    run_dir_abs="$(cd "${run_dir}" && pwd)"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"
    sampler_overrides=()
    if [[ -n "${EVAL_SAMPLER}" ]]; then
        sampler_overrides+=(model.processor.sampler="${EVAL_SAMPLER}")
    fi
    if [[ -n "${EVAL_SAMPLER_STEPS}" ]]; then
        sampler_overrides+=(model.processor.sampler_steps="${EVAL_SAMPLER_STEPS}")
    fi
    if [[ -n "${EVAL_SAMPLER_ORDER}" ]]; then
        sampler_overrides+=(++model.processor.sampler_order="${EVAL_SAMPLER_ORDER}")
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting RB LoLA diffusion ViT-large cached-latent metrics-only eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.mode: encode_once"
        if [[ -n "${EVAL_SAMPLER}" ]]; then
            echo "  model.processor.sampler: ${EVAL_SAMPLER}"
        else
            echo "  model.processor.sampler: inherited from run config"
        fi
        if [[ -n "${EVAL_SAMPLER_STEPS}" ]]; then
            echo "  model.processor.sampler_steps: ${EVAL_SAMPLER_STEPS}"
        fi
        if [[ -n "${EVAL_SAMPLER_ORDER}" ]]; then
            echo "  model.processor.sampler_order: ${EVAL_SAMPLER_ORDER}"
        fi
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.chunk_size: ${EVAL_CHUNK_SIZE}"
        echo "  eval.rollout_start: ${EVAL_ROLLOUT_START}"
        echo "  eval.batch_indices: []"
        echo "  eval.compute_rollout_metrics: true"
        echo "  eval.compute_rollout_coverage: true"
        echo "  hydra.launcher.mem: ${SLURM_MEM}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint=processor.ckpt \
            eval.mode=encode_once \
            "${sampler_overrides[@]}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            +eval.chunk_size="${EVAL_CHUNK_SIZE}" \
            eval.rollout_start="${EVAL_ROLLOUT_START}" \
            eval.transpose_spatial=true \
            'eval.batch_indices=[]' \
            eval.save_rollout_snapshots=false \
            eval.compute_rollout_metrics=true \
            eval.compute_rollout_coverage=true \
            eval.compute_rollout_autoencoded_target_metrics=false \
            eval.benchmark.enabled=false \
            eval.benchmark_rollout.enabled=false \
            hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
            hydra.launcher.additional_parameters.mem="${SLURM_MEM}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
