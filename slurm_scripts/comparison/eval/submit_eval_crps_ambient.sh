#!/bin/bash

set -euo pipefail
# Evaluate CRPS-in-ambient EPD runs trained on 2026-04-24.
# Covers the 4 primary permute_concat runs. CNS ablation-only reruns stay under
# slurm_scripts/ablations/ until they are promoted into the comparison suite.
# All are EPD checkpoints (encoder_processor_decoder.ckpt); eval uses the
# resolved_config.yaml written alongside each run, so the trained architecture
# is reproduced exactly for eval.
#
# Force eval.mode=ambient. These stateless EPD checkpoints can look
# processor-only to eval.mode=auto because PermuteConcat / ChannelsLast add no
# encoder_decoder.* weights, but raw-space ambient rollout is the right route.
#
# Batch size: CRPS eval fits 8/GPU comfortably (ambient 64x64, n_members=10,
# single forward pass per rollout step — no ODE).
#
# We also pin eval.n_members explicitly here so the comparison scripts do not
# depend on the global eval default staying at 10.

EVAL_BATCH_SIZE=8
EVAL_N_MEMBERS=10
TIMEOUT_MIN=240
RUN_DRY_STATES=("true" "false")
EVAL_METRICS="[mse,mae,nmse,nmae,rmse,nrmse,vmse,vrmse,linf,psrmse,psrmse_low,psrmse_mid,psrmse_high,psrmse_tail,pscc,pscc_low,pscc_mid,pscc_high,pscc_tail,crps,fcrps,afcrps,energy,ssr,winkler]"

# Run dirs (absolute paths work; relative paths resolved from repo root).
RUN_DIRS=(
    "outputs/2026-04-24/crps_gs64_vit_azula_large_bed4611_828a161"
    "outputs/2026-04-24/crps_gpe64_vit_azula_large_bed4611_e0a6df5"
    "outputs/2026-04-24/crps_cns64_vit_azula_large_bed4611_c99f534"
    "outputs/2026-04-24/crps_ad64_vit_azula_large_bed4611_da01a04"
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

        echo "Submitting CRPS-ambient eval"
        echo "  mode: ${run_label}"
        echo "  run_dir: ${run_dir}"
        echo "  eval.mode: ambient"
        echo "  eval.batch_size: ${EVAL_BATCH_SIZE}"
        echo "  eval.n_members: ${EVAL_N_MEMBERS}"
        echo "  eval.metrics: ${EVAL_METRICS}"

        uv run autocast eval --mode slurm "${dry_run_arg[@]}" \
            --workdir "${run_dir}" \
            eval.mode=ambient \
            eval.metrics="${EVAL_METRICS}" \
            eval.batch_size="${EVAL_BATCH_SIZE}" \
            eval.n_members="${EVAL_N_MEMBERS}" \
            hydra.launcher.timeout_min="${TIMEOUT_MIN}"
    done
done
