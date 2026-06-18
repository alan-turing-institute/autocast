#!/bin/bash

set -euo pipefail

# CNS-only AE-compression eval: FM-in-latent run on f8 (8x8x8) cached latents.
#
# Eval uses encode_once so metrics stay in raw data space after decoding while
# avoiding the ambient path's per-step decode->encode loop.
#
# Output goes to <run>/eval_encode_once/.

RUN_ID="fm_f8_cns"
RUN_DIR="outputs/2026-05-01/diff_cns64_flow_matching_vit_43cbdde_bb55197"
AE_CKPT="$HOME/autocast/outputs/2026-04-26/ae_cns64_de1b4b7_e1059d7/autoencoder.ckpt"

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=360
EVAL_SUBDIR="eval_encode_once"
ROLLOUT_SNAPSHOT_TIMESTEPS="[0,4,12,30,99]"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Missing run_dir at ${RUN_DIR}" >&2
    exit 1
fi
run_dir_abs="$(realpath "${RUN_DIR}")"

if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
    echo "Missing resolved_config.yaml under ${RUN_DIR}" >&2
    exit 1
fi

eval_ckpt="${run_dir_abs}/processor.ckpt"
if [[ ! -f "${eval_ckpt}" ]]; then
    echo "Missing processor.ckpt under ${RUN_DIR}" >&2
    exit 1
fi
eval_ckpt_abs="$(realpath "${eval_ckpt}")"

if [[ ! -f "${AE_CKPT}" ]]; then
    echo "Missing AE checkpoint at ${AE_CKPT}" >&2
    exit 1
fi
ae_ckpt_abs="$(realpath "${AE_CKPT}")"

eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting f8 FM encode_once eval"
    echo "  mode: ${run_label}"
    echo "  run_id: ${RUN_ID}"
    echo "  run_dir: ${run_dir_abs}"
    echo "  eval.checkpoint: ${eval_ckpt_abs}"
    echo "  autoencoder_checkpoint: ${ae_ckpt_abs}"
    echo "  eval.mode: encode_once"
    echo "  output_subdir: ${EVAL_SUBDIR}"

    uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
        --workdir "${run_dir_abs}" \
        --output-subdir "${EVAL_SUBDIR}" \
        eval.checkpoint="${eval_ckpt_abs}" \
        eval.mode=encode_once \
        +autoencoder_checkpoint="${ae_ckpt_abs}" \
        eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
        eval.video_dir="${eval_output_dir}/videos" \
        eval.save_rollout_snapshots=true \
        eval.rollout_snapshot_dir="${eval_output_dir}/videos/snapshots" \
        eval.rollout_snapshot_timesteps="${ROLLOUT_SNAPSHOT_TIMESTEPS}" \
        eval.rollout_snapshot_format=png \
        eval.metrics="${EVAL_METRICS}" \
        eval.batch_size="${EVAL_BATCH_SIZE}" \
        eval.n_members="${EVAL_N_MEMBERS}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}"
done
