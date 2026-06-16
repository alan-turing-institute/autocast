#!/bin/bash

set -euo pipefail
# Evaluate the RB effective-batch 24h FM cached-latent run
# (the_well_rayleigh_benard_effbatch24h_fm_latent_b256) with
# eval.mode=encode_once.
#
# This is a processor-only flow-matching ViT trained on cached LoLA latents.
# There is no Lightning autoencoder.ckpt: the LoLA autoencoder is discovered
# automatically by the eval pipeline by walking up from datamodule.data_path
# until it finds config.yaml + state.pth (here:
# datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/). The
# wrapped encoder/decoder carry their own mean/std internally, and the eval
# pipeline substitutes the cached-latents datamodule for an unnormalized raw
# TheWellDataModule. See slurm_scripts/comparison/the_well/
# submit_eval_rayleigh_benard_fm_vit.sh, HANDOFF.md and src/autocast/external/lola/.
#
# Batch size: RB ambient resolution is 512x128 (vs 64x64 for the Well-2D basis),
# and encode_once still pays a full decoder pass on the entire rollout horizon
# (max_rollout_steps=25 -> up to 100 frames per sample). FM additionally
# integrates flow_ode_steps per rollout step, so it is the most expensive of the
# three combinations. EVAL_BATCH_SIZE=1 with EVAL_CHUNK_SIZE=8 keeps per-stage
# activation tensors small; chunking is numerically identical to the non-chunked
# LoLA forward.
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
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-8}"
EVAL_DIAGNOSTIC_MEMBER_INDICES="${EVAL_DIAGNOSTIC_MEMBER_INDICES:-[0]}"
EVAL_ROLLOUT_MEMBER_RENDER_MODE="${EVAL_ROLLOUT_MEMBER_RENDER_MODE:-both}"
# LOLA RB paper figures use start=16 as the final conditioning timestep.
EVAL_ROLLOUT_START="${EVAL_ROLLOUT_START:-16}"
EVAL_MAX_ROLLOUT_STEPS="${EVAL_MAX_ROLLOUT_STEPS:-46}"
EVAL_SUBDIR="${EVAL_SUBDIR:-eval_encode_once_start${EVAL_ROLLOUT_START}}"
EVAL_AUTOENCODED_TARGET_METRICS="${EVAL_AUTOENCODED_TARGET_METRICS:-true}"
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
    "outputs/2026-05-19/the_well_rayleigh_benard_effbatch24h_fm_latent_b256"
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
    if [[ ! -f "${run_dir}/processor.ckpt" ]]; then
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

        echo "Submitting RB eff-batch 24h FM cached-latent eval (mode=encode_once, LoLA AE)"
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
            eval.benchmark.enabled=false \
            eval.benchmark_rollout.enabled=false \
            hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
            hydra.launcher.additional_parameters.mem="${SLURM_MEM}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
