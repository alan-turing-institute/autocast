#!/bin/bash

set -euo pipefail
# Ambient eval submitter for the 2026-04-24 CRPS ensemble-size runs.
#
# This variant selects the best multi-Winkler checkpoint after the 0.25
# progress cutoff: best-multiwinkler-from0p25-*.ckpt. The standard ensemble
# eval submitter under ../eval/ is left unchanged.
#
# Force eval.mode=ambient. These stateless EPD checkpoints can look
# processor-only to eval.mode=auto because PermuteConcat / ChannelsLast add no
# encoder_decoder.* weights, but raw-space ambient rollout is the right route.
#
# Batch size: keep 4/GPU as the conservative first pass used by the standard
# ambient ensemble-size eval script.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
EVAL_SUBDIR="eval_best_multiwinkler_from0p25"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    "outputs/2026-04-24/ensemble_size/crps_gs64_vit_azula_large_bed4611_4d04729"
    "outputs/2026-04-24/ensemble_size/crps_gpe64_vit_azula_large_bed4611_6b78265"
    "outputs/2026-04-24/ensemble_size/crps_cns64_vit_azula_large_bed4611_5758ebc"
    "outputs/2026-04-24/ensemble_size/crps_ad64_vit_azula_large_bed4611_69c99bf"
)

resolve_multiwinkler_checkpoint() {
    local run_dir="$1"
    local -a ckpts=()

    mapfile -t ckpts < <(
        find "${run_dir}" -type f -path '*/checkpoints/best-multiwinkler-from0p25-*.ckpt' | sort
    )

    if (( ${#ckpts[@]} >= 1 )); then
        printf '%s\n' "${ckpts[$(( ${#ckpts[@]} - 1 ))]}"
        return 0
    fi

    return 1
}

for run_dir in "${RUN_DIRS[@]}"; do
    run_dir_abs="$(realpath "${run_dir}")"
    if [[ ! -f "${run_dir_abs}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi

    if ! eval_ckpt="$(resolve_multiwinkler_checkpoint "${run_dir_abs}")"; then
        echo "Skipping ${run_dir}: best-multiwinkler-from0p25-*.ckpt missing" >&2
        continue
    fi
    eval_ckpt_abs="$(realpath "${eval_ckpt}")"
    eval_output_dir="${run_dir_abs}/${EVAL_SUBDIR}"

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting ensemble-size CRPS ambient eval (best multi-Winkler from 0.25)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir_abs}"
        echo "  eval.checkpoint: ${eval_ckpt_abs}"
        echo "  eval.mode: ambient"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir_abs}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt_abs}" \
            eval.mode=ambient \
            eval.csv_path="${eval_output_dir}/evaluation_metrics.csv" \
            eval.video_dir="${eval_output_dir}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
