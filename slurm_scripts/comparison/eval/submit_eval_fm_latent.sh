#!/bin/bash

set -euo pipefail
# Evaluate FM cached-latent processor runs from 2026-04-20 using the default
# eval.mode=auto path.
#
# After PR #339, auto resolves to encode_once for processor-only cached-latent
# runs when autoencoder_checkpoint is supplied. That keeps metrics in raw data
# space against decoded rollouts, without charging per-step decode->encode
# drift as the ambient ablation does.
#
# Batch size: encode_once is cheaper than ambient because the encoder only
# runs once, but FM still pays 50 ODE substeps per rollout step. 4/GPU remains
# a safe baseline and stays aligned with the ambient FM comparison script.

EVAL_BATCH_SIZE=4
TIMEOUT_MIN=360
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"
    "outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"
    "outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"
    "outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-20/diff_gs64_flow_matching_vit_09490da_7e9e331"]="$HOME/autocast/outputs/2026-04-17/ae_gs64_3a7999b_ed36b8e/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_gpe64_flow_matching_vit_09490da_47bf39a"]="$HOME/autocast/outputs/2026-04-17/ae_gpe64_3a7999b_31e1c9f/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_cns64_flow_matching_vit_09490da_636fcc3"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
    ["outputs/2026-04-20/diff_ad64_flow_matching_vit_09490da_dae1382"]="$HOME/autocast/outputs/2026-04-17/ae_ad64_3a7999b_1a1e300/autoencoder.ckpt"
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

        echo "Submitting FM cached-latent eval (mode=auto -> encode_once)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  autoencoder_checkpoint: ${ae_ckpt}"
        echo "  eval.mode: auto"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.checkpoint=processor.ckpt \
            autoencoder_checkpoint="${ae_ckpt}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
