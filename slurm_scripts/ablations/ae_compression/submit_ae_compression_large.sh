#!/bin/bash

set -euo pipefail
# CNS-only AE compression ablation — 24h production run.
# Uses COSINE_EPOCHS=512 to match the main-study AE schedule
# (slurm_scripts/comparison/ae/submit_ae_large.sh), so AE comparisons are
# on equal-schedule footing. CNS at 91 s/ep used ~58% of its 24h budget at
# 512 epochs in the baseline; the 4-level AE adds modest compute (one extra
# shallow level, narrower widths) and should remain comfortably within 24h.
#
# If the run wall-clocks out, fall back to submit_ae_compression_timing.sh
# to derive a safe per-budget value.
COSINE_EPOCHS=512

BUDGET_MAX_TIME="01:00:00:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

datamodule="conditioned_navier_stokes"
experiment="ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting AE compression production run"
    echo "  mode: ${run_label}"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  cosine_epochs: ${COSINE_EPOCHS}"

    uv run autocast ae --mode slurm "${dry_run_arg[@]}" \
        datamodule="${datamodule}" \
        local_experiment="${experiment}" \
        logging.wandb.enabled=true \
        +optimizer.cosine_epochs="${COSINE_EPOCHS}" \
        hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
        trainer.max_time="${BUDGET_MAX_TIME}" \
        +trainer.max_epochs="${COSINE_EPOCHS}"
done
