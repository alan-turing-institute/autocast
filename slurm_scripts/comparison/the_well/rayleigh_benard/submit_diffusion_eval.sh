#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"

if (($# == 0)) || [[ "${1:-}" == "all" ]]; then
    EVAL_KEYS=(euler ab_o3 ddpm)
else
    EVAL_KEYS=("$@")
fi

RUN_GLOB="${RUN_GLOB:-outputs/*/the_well_rayleigh_benard_diffusion_vit_large_lola4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-8}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
EVAL_ROLLOUT_START="${EVAL_ROLLOUT_START:-16}"
EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-true}"
EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-true}"
EVAL_AUTOENCODED_TARGET_METRICS="${EVAL_AUTOENCODED_TARGET_METRICS:-true}"
EVAL_METRICS="${EVAL_METRICS:-${RB_EVAL_METRICS_DEFAULT}}"
EVAL_SAMPLER_OVERRIDE="${EVAL_SAMPLER:-}"
EVAL_SAMPLER_STEPS_OVERRIDE="${EVAL_SAMPLER_STEPS:-}"
EVAL_SAMPLER_ORDER_OVERRIDE="${EVAL_SAMPLER_ORDER:-}"
EVAL_N_MEMBERS_OVERRIDE="${EVAL_N_MEMBERS:-}"
EVAL_SUBDIR_OVERRIDE="${EVAL_SUBDIR:-}"
TIMEOUT_MIN="${TIMEOUT_MIN:-1439}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
SLURM_MEM="${SLURM_MEM:-115G}"

find_run_dirs() {
    local matches=()
    local match

    shopt -s nullglob
    matches=( ${RUN_GLOB} )
    shopt -u nullglob

    if ((${#matches[@]} == 0)); then
        echo "No matching Rayleigh-Benard diffusion runs found for RUN_GLOB=${RUN_GLOB}" >&2
        exit 1
    fi

    for match in "${matches[@]}"; do
        if [[ -f "${match}/resolved_config.yaml" && -e "${match}/processor.ckpt" ]]; then
            printf '%s\n' "${match}"
        else
            echo "Skipping ${match}: resolved_config.yaml or processor.ckpt missing" >&2
        fi
    done
}

set_eval_defaults() {
    local key="$1"

    EXTRA_OVERRIDES=()

    case "${key}" in
        euler)
            METHOD_LABEL="Euler probability-flow ODE"
            EVAL_SAMPLER="${EVAL_SAMPLER_OVERRIDE:-euler}"
            EVAL_SAMPLER_STEPS="${EVAL_SAMPLER_STEPS_OVERRIDE:-50}"
            EVAL_N_MEMBERS="${EVAL_N_MEMBERS_OVERRIDE:-10}"
            EVAL_SUBDIR="${EVAL_SUBDIR_OVERRIDE:-eval_encode_once_euler${EVAL_SAMPLER_STEPS}_start${EVAL_ROLLOUT_START}}"
            ;;
        ab_o3)
            METHOD_LABEL="Adams-Bashforth order 3"
            EVAL_SAMPLER="${EVAL_SAMPLER_OVERRIDE:-ab}"
            EVAL_SAMPLER_STEPS="${EVAL_SAMPLER_STEPS_OVERRIDE:-16}"
            EVAL_SAMPLER_ORDER="${EVAL_SAMPLER_ORDER_OVERRIDE:-3}"
            EVAL_N_MEMBERS="${EVAL_N_MEMBERS_OVERRIDE:-16}"
            EVAL_SUBDIR="${EVAL_SUBDIR_OVERRIDE:-eval_encode_once_ab${EVAL_SAMPLER_STEPS}_o${EVAL_SAMPLER_ORDER}_start${EVAL_ROLLOUT_START}}"
            EXTRA_OVERRIDES=("++model.processor.sampler_order=${EVAL_SAMPLER_ORDER}")
            ;;
        ddpm)
            METHOD_LABEL="DDPM stochastic reverse sampler"
            EVAL_SAMPLER="${EVAL_SAMPLER_OVERRIDE:-ddpm}"
            EVAL_SAMPLER_STEPS="${EVAL_SAMPLER_STEPS_OVERRIDE:-50}"
            EVAL_N_MEMBERS="${EVAL_N_MEMBERS_OVERRIDE:-10}"
            EVAL_SUBDIR="${EVAL_SUBDIR_OVERRIDE:-eval_encode_once_ddpm${EVAL_SAMPLER_STEPS}_start${EVAL_ROLLOUT_START}}"
            ;;
        *)
            echo "Unknown diffusion eval key: ${key}" >&2
            exit 1
            ;;
    esac
}

submit_one_eval() {
    local key="$1"
    local run_dir="$2"
    local run_dry="$3"
    local mode_label
    local run_dir_abs
    local eval_output_dir

    set_eval_defaults "${key}"

    mode_label="$(rb_print_mode "${run_dry}")"
    run_dir_abs="$(cd "${run_dir}" && pwd)"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    echo "Submitting RB LOLA diffusion eval"
    echo "  mode: ${mode_label}"
    echo "  method: ${METHOD_LABEL}"
    echo "  run_dir: ${run_dir}"
    echo "  output_subdir: ${EVAL_SUBDIR}"
    echo "  model.processor.sampler: ${EVAL_SAMPLER}"
    echo "  model.processor.sampler_steps: ${EVAL_SAMPLER_STEPS}"
    echo "  eval.n_members: ${EVAL_N_MEMBERS}"
    echo "  eval.rollout_start: ${EVAL_ROLLOUT_START}"

    rb_submit eval "${run_dry}" \
        --workdir "${run_dir}" \
        --output-subdir "${EVAL_SUBDIR}" \
        "eval.checkpoint=processor.ckpt" \
        "eval.mode=encode_once" \
        "model.processor.sampler=${EVAL_SAMPLER}" \
        "model.processor.sampler_steps=${EVAL_SAMPLER_STEPS}" \
        "eval.csv_path=${eval_output_dir}/evaluation_metrics.csv" \
        "eval.video_dir=${eval_output_dir}/videos" \
        "eval.metrics=${EVAL_METRICS}" \
        "eval.batch_size=${EVAL_BATCH_SIZE}" \
        "eval.n_members=${EVAL_N_MEMBERS}" \
        "+eval.chunk_size=${EVAL_CHUNK_SIZE}" \
        "eval.rollout_start=${EVAL_ROLLOUT_START}" \
        "eval.transpose_spatial=true" \
        "eval.compute_rollout_autoencoded_target_metrics=${EVAL_AUTOENCODED_TARGET_METRICS}" \
        "eval.rollout_member_indices=${EVAL_DIAGNOSTIC_MEMBER_INDICES}" \
        "eval.rollout_member_render_mode=${EVAL_ROLLOUT_MEMBER_RENDER_MODE}" \
        "eval.benchmark.enabled=${EVAL_BENCHMARK_ENABLED}" \
        "eval.benchmark_rollout.enabled=${EVAL_BENCHMARK_ROLLOUT_ENABLED}" \
        "hydra.launcher.cpus_per_task=${CPUS_PER_TASK}" \
        "hydra.launcher.additional_parameters.mem=${SLURM_MEM}" \
        "hydra.launcher.timeout_min=${TIMEOUT_MIN}" \
        "${EXTRA_OVERRIDES[@]}"
}

mapfile -t RUN_DIRS < <(find_run_dirs)
if ((${#RUN_DIRS[@]} == 0)); then
    echo "No usable Rayleigh-Benard diffusion runs found." >&2
    exit 1
fi

for key in "${EVAL_KEYS[@]}"; do
    for run_dir in "${RUN_DIRS[@]}"; do
        while IFS= read -r run_dry; do
            submit_one_eval "${key}" "${run_dir}" "${run_dry}"
        done < <(rb_run_dry_states)
    done
done
