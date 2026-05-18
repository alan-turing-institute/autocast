#!/bin/bash

set -euo pipefail
# Evaluate the 2026-05-12 Rayleigh-Benard FM ViT runs (small + large) in the
# *cached LoLA latent space* with eval.mode=latent and
# eval.latent_space_metrics=true.
#
# Companion to submit_eval_rayleigh_benard_fm_vit.sh, which scores the same
# runs in ambient space via eval.mode=encode_once. Together the two scripts
# give CRPS in both spaces for the same checkpoints.
#
# eval.mode=latent + eval.latent_space_metrics=true skips the decoder
# entirely: predictions and ground truth are both kept in the cached-latent
# tensor space, and the metric stack (CRPS etc.) runs against the raw latent.
# Implications:
#   - No decoder activations -> batch size can be larger than the ambient
#     script (which is pinned to EVAL_BATCH_SIZE=1 for the 512x128 decoder
#     pass). Latent rollouts are tiny (LoLA dcae_f32c64 -> 16x4x64 per frame),
#     so EVAL_BATCH_SIZE=4 with EVAL_N_MEMBERS=10 is comfortable.
#   - +eval.chunk_size is unused (no encoder/decoder forward), so omitted.
#   - Output lives at <run_dir>/eval_latent/ via --output-subdir, so it does
#     not overwrite the ambient eval/ produced by the sibling script.
#   - psrmse*/pscc*/variogram-style physics-aware metrics are not meaningful
#     in latent space; the eval pipeline emits a warning. We still request
#     the full metric list to mirror the ambient script and keep columns
#     aligned in downstream analysis -- ignore those columns in latent.
#   - transpose_spatial only affects rollout snapshot rendering; we keep it
#     consistent with the ambient script.
#
# Single-GPU on purpose, same rationale as the ambient script: matches the
# comparison/eval/ baseline, avoids the DDP tail-padding bias for aggregate
# metrics, and keeps rollout rendering deterministic.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,vmse_v2,vrmse_v2,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_PATTERNS=(
    "outputs/2026-05-12/diff_rayleigh_benard_flow_matching_vit_5ee7659_1a0afcd"
    "outputs/2026-05-12/diff_rayleigh_benard_flow_matching_vit_4d8cf74_bd6a7ae"
    "outputs/*/diff_rayleigh_benard_flow_matching_masked_window_vit_*"
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

        echo "Submitting RB FM cached-latent eval (mode=latent, latent_space_metrics=true)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.mode: latent"
        echo "  eval.latent_space_metrics: true"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.transpose_spatial: true"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir eval_latent \
            eval.checkpoint=processor.ckpt \
            eval.mode=latent \
            eval.latent_space_metrics=true \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            eval.transpose_spatial=true \
            hydra.launcher.cpus_per_task=8 \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
