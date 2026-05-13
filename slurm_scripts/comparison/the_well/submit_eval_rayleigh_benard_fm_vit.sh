#!/bin/bash

set -euo pipefail
# Evaluate the 2026-05-12 Rayleigh-Benard FM ViT runs (small + large) using
# eval.mode=encode_once.
#
# These runs are processor-only flow-matching ViTs trained on cached LoLA
# latents. There is no Lightning autoencoder.ckpt: the LoLA autoencoder is
# discovered automatically by the eval pipeline by walking up from
# datamodule.data_path until it finds config.yaml + state.pth. The wrapped
# encoder/decoder carry their own mean/std internally, and the eval pipeline
# substitutes the cached-latents datamodule for an unnormalized raw
# TheWellDataModule. See HANDOFF.md and src/autocast/external/lola/.
#
# Batch size: RB ambient resolution is 512x128 (vs 64x64 for the Well-2D
# basis), and encode_once still pays a full decoder pass on the entire rollout
# horizon (max_rollout_steps=25, stride=4 -> up to 100 frames per sample).
# With n_members=10 and flow_ode_steps=50 through the processor, EVAL_BATCH_SIZE=2
# is the safe starting point; drop to 1 on OOM, raise toward 4 if comfortable.
#
# Single-GPU on purpose: matches the comparison/eval/ scripts (known-good
# wall-clock and outputs), avoids the DDP tail-padding bias for aggregate
# metrics, and keeps `_render_rollouts` correct -- under DDP the renderer is
# called on every rank and writes to the same `video_dir/batch_<idx>_sample_<i>`
# paths from each rank's local sample offset, racing on file output. If the
# first run is too slow, follow up by splitting into a 4-GPU metrics-only job
# (`+distributed=ddp_4gpu_slurm eval.batch_indices=[]`) plus a 1-GPU
# render-only job, rather than turning DDP on here.

EVAL_BATCH_SIZE=2
EVAL_N_MEMBERS=10
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-05-12/diff_rayleigh_benard_flow_matching_vit_5ee7659_1a0afcd"
    "outputs/2026-05-12/diff_rayleigh_benard_flow_matching_vit_4d8cf74_bd6a7ae"
)

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

        echo "Submitting RB FM cached-latent eval (mode=encode_once, LoLA AE)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.mode: encode_once"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.checkpoint=processor.ckpt \
            eval.mode=encode_once \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.cpus_per_task=8 \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
