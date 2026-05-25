#!/bin/bash

set -euo pipefail
# Evaluate the Rayleigh-Benard LoLA diffusion ViT-large cached-latent run with
# LoLA's *inference* sampler instead of Euler, to confirm whether the eval/SSR
# result depends on the sampler.
#
# Background: the trained run sampled with deterministic Euler (50 steps); LoLA
# evaluates with algorithm="ab" (Adams-Bashforth multistep, order=2) at 16
# steps. AB is also a *deterministic* ODE integrator, so this isolates the
# integrator (Euler vs higher-order multistep) on the stiff VE probability-flow
# ODE -- the variable the earlier "AB ~= Euler" flow-matching test could not
# probe (FM's OT path is near-straight, so integrator order barely matters
# there).
#
# Nothing is retrained: the sampler is a processor attribute set at construction
# from resolved_config.yaml, so we just override model.processor.sampler /
# .sampler_steps at eval time (struct-safe; both keys exist in the resolved
# config). sampler_order defaults to 2 (LoLA) in the DiffusionProcessor ctor.
#
# Outputs go to a distinct subdir (eval_encode_once_ab16) with explicit
# csv_path/video_dir so this does NOT clobber the Euler eval (eval_encode_once).
#
# Same single-GPU rationale as submit_eval_rayleigh_benard_diffusion_vit_large_lola.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

# Sampler knobs. LoLA's eval recipe is ab / 16 steps. Set EVAL_SAMPLER_STEPS=50
# to control for step count against the Euler-50 baseline instead.
EVAL_SAMPLER="${EVAL_SAMPLER:-ab}"
EVAL_SAMPLER_STEPS="${EVAL_SAMPLER_STEPS:-16}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
# Baseline Euler eval used 10 members; keep 10 for a clean sampler-only diff.
# (LoLA itself used 16 members -- set EVAL_N_MEMBERS=16 to also match that.)
EVAL_N_MEMBERS="${EVAL_N_MEMBERS:-10}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-8}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_encode_once_${EVAL_SAMPLER}${EVAL_SAMPLER_STEPS}}"
EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-true}"
EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-true}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_MEM="${SLURM_MEM:-115G}"
DRY_RUN_ONLY="${DRY_RUN_ONLY:-false}"
if [[ "${DRY_RUN_ONLY}" == "true" ]]; then
    RUN_DRY_STATES=("true")
else
    RUN_DRY_STATES=("true" "false")
fi
EVAL_METRICS="${EVAL_METRICS:-[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,vmse_v2,vrmse_v2,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]}"

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

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting RB LoLA diffusion ViT-large cached-latent eval (AB sampler)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  model.processor.sampler: ${EVAL_SAMPLER} (order=2)"
        echo "  model.processor.sampler_steps: ${EVAL_SAMPLER_STEPS}"
        echo "  eval.mode: encode_once"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.chunk_size: ${EVAL_CHUNK_SIZE}"
        echo "  eval.transpose_spatial: true"
        echo "  eval.benchmark.enabled: ${EVAL_BENCHMARK_ENABLED}"
        echo "  eval.benchmark_rollout.enabled: ${EVAL_BENCHMARK_ROLLOUT_ENABLED}"
        echo "  hydra.launcher.mem: ${SLURM_MEM}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint=processor.ckpt \
            eval.mode=encode_once \
            model.processor.sampler="${EVAL_SAMPLER}" \
            model.processor.sampler_steps="${EVAL_SAMPLER_STEPS}" \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            +eval.chunk_size="${EVAL_CHUNK_SIZE}" \
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
