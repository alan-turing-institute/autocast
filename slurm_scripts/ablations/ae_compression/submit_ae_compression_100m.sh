#!/bin/bash

set -euo pipefail
# CNS f8 capacity sweep — ~100M params (99M, widths [90,180,360,720]).
# 256 epochs to stay within the 24h wall: 99M ~= 1.98x FLOPs/epoch vs the
# 50M run (which used ~14h at 512 epochs), so 256 epochs ~= 14h linear,
# allowing for 1.3x empirical slowdown still leaves comfortable margin.
COSINE_EPOCHS=256

BUDGET_MAX_TIME="01:00:00:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

datamodule="conditioned_navier_stokes"
experiment="ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8_100m"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting AE compression 100M run"
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
