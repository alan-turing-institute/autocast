#!/bin/bash

set -euo pipefail
# Ambient eval submitter for the 75%-schedule checkpoints from the current
# CRPS ensemble-size runs.
#
# This mirrors ../eval/submit_eval_crps_ambient.sh but keeps outputs under a
# sibling eval_0p75/ folder so partial-schedule metrics, rollout videos, and
# SLURM logs do not mix with the standard final-checkpoint evals.
#
# The large-run training scripts save checkpoints at 25/50/75/100% of the
# cosine schedule via quarter-*.ckpt filenames. Rather than hard-coding epoch
# numbers per dataset, we sort the available quarter checkpoints and pick the
# third one (the 0.75 checkpoint) for each run.
#
# Batch size: keep 4/GPU as the same conservative first pass used by the
# standard ambient ensemble-size eval script.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
EVAL_SUBDIR="eval_0p75"
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    # CNS pilot runs retained alongside the compute-matched sweep.
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_5e157a5"  # ensemble_m16_fixed_bs32
    # Compute-matched eff_bs1024 runs across comparison datasets.
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_dcd79e4"
    "outputs/2026-04-21/ensemble_size/crps_gs64_vit_azula_large_ac1bb06_639963f"
    "outputs/2026-04-21/ensemble_size/crps_gpe64_vit_azula_large_ac1bb06_638585e"
    "outputs/2026-04-21/ensemble_size/crps_ad64_vit_azula_large_ac1bb06_ef6368d"
)

resolve_three_quarter_checkpoint() {
    local run_dir="$1"
    local -a quarter_ckpts=()

    mapfile -t quarter_ckpts < <(
        find "${run_dir}" -maxdepth 1 -type f -name 'quarter-*.ckpt' | sort
    )

    if (( ${#quarter_ckpts[@]} < 3 )); then
        return 1
    fi

    printf '%s\n' "${quarter_ckpts[2]}"
}

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi

    if ! eval_ckpt="$(resolve_three_quarter_checkpoint "${run_dir}")"; then
        echo "Skipping ${run_dir}: fewer than three quarter-*.ckpt files found" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting ensemble-size CRPS ambient eval (0.75 checkpoint)"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.checkpoint: ${eval_ckpt}"
        echo "  output_subdir: ${EVAL_SUBDIR}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            --output-subdir "${EVAL_SUBDIR}" \
            eval.checkpoint="${eval_ckpt}" \
            eval.csv_path="${run_dir}/${EVAL_SUBDIR}/evaluation_metrics.csv" \
            eval.video_dir="${run_dir}/${EVAL_SUBDIR}/videos" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
