#!/bin/bash

set -euo pipefail
# Ambient eval submitter for the current CRPS ensemble-size runs.
#
# This lives under the ablation directory because the run set is still
# study-specific: the CNS fixed_bs32 pilot remains an ablation, while the
# eff_bs1024 runs may later be promoted into the canonical comparison suite.
# If that promotion happens, move the promoted run dirs into
# slurm_scripts/comparison/eval/ and keep ablation-only evals local here.
#
# We keep eval.n_members fixed at 10 to match the current comparison-study
# eval regime, even though these checkpoints were trained with model.n_members=16.
# That keeps the eval sampling budget comparable across studies unless we
# intentionally decide to benchmark full m=16 rollout at eval time.
#
# Batch size: keep 4/GPU as a conservative first pass for these m=16 ablation
# checkpoints. If cluster headroom is clearly available, try 8 later.

EVAL_BATCH_SIZE=4
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

RUN_DIRS=(
    # CNS pilot runs from the original part-day sweep.
    # "outputs/2026-04-20_part_8hr/ensemble_size/crps_cns64_vit_azula_large_0db40e1_5e157a5"  # ensemble_m16_fixed_bs32
    # "outputs/2026-04-20_part_8hr/ensemble_size/crps_cns64_vit_azula_large_0db40e1_dcd79e4"  # ensemble_m16_eff_bs1024
    # Active compute-matched eff_bs1024 runs on the other comparison datasets.
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_5e157a5"  # ensemble_m16_fixed_bs32
    "outputs/2026-04-20/ensemble_size/crps_cns64_vit_azula_large_0db40e1_dcd79e4"  # ensemble_m16_eff_bs1024
    # "outputs/2026-04-21/ensemble_size/crps_gs64_vit_azula_large_ac1bb06_639963f"
    # "outputs/2026-04-21/ensemble_size/crps_gpe64_vit_azula_large_ac1bb06_638585e"
    # "outputs/2026-04-21/ensemble_size/crps_ad64_vit_azula_large_ac1bb06_ef6368d"
)

for run_dir in "${RUN_DIRS[@]}"; do
    if [[ ! -f "${run_dir}/resolved_config.yaml" ]]; then
        echo "Skipping ${run_dir}: resolved_config.yaml missing" >&2
        continue
    fi

    for run_dry in "${RUN_DRY_STATES[@]}"; do
        dry_run_arg=()
        run_label="slurm"
        if [[ "${run_dry}" == "true" ]]; then
            dry_run_arg=(--dry-run)
            run_label="slurm --dry-run"
        fi

        echo "Submitting ensemble-size CRPS ambient eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
