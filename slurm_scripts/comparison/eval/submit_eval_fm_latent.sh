#!/bin/bash

set -euo pipefail
# Evaluate FM cached-latent processor runs (2026-04-18) in AMBIENT mode.
#
# eval.mode=ambient forces encoder->processor->decoder rollout at every
# step, so decode/encode drift is included in the metrics — the apples-to-
# apples regime for comparison with the ambient FM baseline.
#
# The eval.mode selector landed via PR #327 and is now in-tree. When ambient
# is requested on a cached-latents datamodule, eval auto-substitutes the raw
# datamodule from <cache_dir>/autoencoder_config.yaml; the trained AE weights
# are supplied via autoencoder_checkpoint.
#
# Batch size: ambient rollout pays encode/decode every step plus 50 ODE
# substeps through the processor. Cached-latent processor forward is lighter
# (64 tokens vs 256 for ambient FM), so 4/GPU is a safe start; the tight
# spot is the same ODE + AE stack so it mirrors FM-ambient.

EVAL_BATCH_SIZE=4
TIMEOUT_MIN=360
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-18/diff_gs64_flow_matching_vit_0f89f06_f6e8f51"
    "outputs/2026-04-18/diff_gpe64_flow_matching_vit_0f89f06_b954f94"
    "outputs/2026-04-18/diff_cns64_flow_matching_vit_0f89f06_0e1c64b"
    "outputs/2026-04-18/diff_ad64_flow_matching_vit_0f89f06_df2137c"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-18/diff_gs64_flow_matching_vit_0f89f06_f6e8f51"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["outputs/2026-04-18/diff_gpe64_flow_matching_vit_0f89f06_b954f94"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    ["outputs/2026-04-18/diff_cns64_flow_matching_vit_0f89f06_0e1c64b"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
    ["outputs/2026-04-18/diff_ad64_flow_matching_vit_0f89f06_df2137c"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
)

for run_dir in "${RUN_DIRS[@]}"; do
    ae_ckpt="${AE_CKPT[$run_dir]:-}"
    if [[ -z "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: no autoencoder_checkpoint mapping" >&2
        continue
    fi
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi
    if [[ ! -f "${ae_ckpt}" ]]; then
        echo "Skipping ${run_dir}: AE checkpoint missing at ${ae_ckpt}" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting FM cached-latent eval (mode=ambient)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  autoencoder_checkpoint: ${ae_ckpt}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.checkpoint=processor.ckpt \
            ++eval.mode=ambient \
            +autoencoder_checkpoint="${ae_ckpt}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
