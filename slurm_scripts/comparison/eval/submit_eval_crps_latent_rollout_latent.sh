#!/bin/bash

set -euo pipefail
# Evaluate CRPS cached-latent processor runs (2026-04-20_part) in LATENT mode.
#
# eval.mode=latent rolls out only in latent space and writes results to
# eval_latent/ so ambient-vs-latent comparisons can coexist per run.
#
# The eval.mode selector landed via PR #327 and is now in-tree. We still pass
# autoencoder_checkpoint to load the trained AE for eval setup/final decode.
#
# Batch size: latent rollout avoids per-step AE encode/decode, so 8/GPU is a
# conservative setting and matches the ambient-compare CRPS script.
#
# We also pin eval.n_members explicitly here so the comparison scripts do not
# depend on the global eval default staying at 10.

EVAL_BATCH_SIZE=8
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
RUN_DRY_STATES=("true" "false")
EVAL_SUBDIR="eval_latent"
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

# (run_dir, autoencoder_checkpoint) pairs. Extend as more cached-latent CRPS
# runs land (gs, gpe, ad) — the AE paths are the same as training.
RUN_DIRS=(
    "outputs/2026-04-20_part/crps_cns64_vit_azula_large_09490da_8b7573d"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-20_part/crps_cns64_vit_azula_large_09490da_8b7573d"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
)
source .venv/bin/activate
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

        echo "Submitting CRPS cached-latent eval (mode=latent)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  autoencoder_checkpoint: ${ae_ckpt}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"
        echo "  output_dir: ${run_dir}/${EVAL_SUBDIR}"

        autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.checkpoint=processor.ckpt \
            ++eval.mode=latent \
            +autoencoder_checkpoint="${ae_ckpt}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.sweep.dir="${run_dir}/${EVAL_SUBDIR}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
