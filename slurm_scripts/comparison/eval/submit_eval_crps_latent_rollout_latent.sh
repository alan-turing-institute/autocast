#!/bin/bash

set -euo pipefail
# Evaluate CRPS cached-latent processor runs (2026-04-19) in AMBIENT mode.
#
# eval.mode=ambient forces encoder->processor->decoder rollout at every
# step, so decode/encode drift is included in the metrics. This makes the
# latent-space CRPS numbers directly comparable with the ambient CRPS and
# FM baselines (see slurm_scripts/comparison/eval/README.md).
#
# Requires PR #327 (origin/add-eval-modes — eval.mode selector). When ambient
# is requested on a cached-latents datamodule, eval auto-substitutes the raw
# datamodule from <cache_dir>/autoencoder_config.yaml; the trained AE weights
# are supplied via autoencoder_checkpoint.
#
# Batch size: cached-latent eval pays the ambient AE encode/decode per step
# but processor forward is cheap (64 tokens vs 256 for ambient-patch4), so
# 8/GPU fits comfortably — same as pure-ambient CRPS.

EVAL_BATCH_SIZE=8
TIMEOUT_MIN=240
RUN_DRY_STATES=("true" "false")

# (run_dir, autoencoder_checkpoint) pairs. Extend as more cached-latent CRPS
# runs land (gs, gpe, ad) — the AE paths are the same as training.
RUN_DIRS=(
    "outputs/2026-04-19/crps_cns64_vit_azula_large_58712c4_71ba7be"
)
declare -A AE_CKPT=(
    ["outputs/2026-04-19/crps_cns64_vit_azula_large_58712c4_71ba7be"]="$HOME/autocast/outputs/2026-04-17/ae_cns64_3a7999b_b9c29f8/autoencoder.ckpt"
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

        echo "Submitting CRPS cached-latent eval (mode=latent)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  autoencoder_checkpoint: ${ae_ckpt}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.checkpoint=processor.ckpt \
            ++eval.mode=latent \
            +autoencoder_checkpoint="${ae_ckpt}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
