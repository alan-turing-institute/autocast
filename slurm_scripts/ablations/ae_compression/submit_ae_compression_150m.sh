#!/bin/bash

set -euo pipefail
# CNS f8 capacity sweep — ~150M params (153M, widths [112,224,448,896]).
# 256 epochs matches the 100M run for an apples-to-apples comparison: 153M
# ~= 3.06x FLOPs/epoch vs the 50M run (~14h at 512 epochs) -> 256 epochs
# ~= 21h linear. May wall-out before full schedule completion under any
# empirical slowdown — checkpoint callbacks preserve the best state.
COSINE_EPOCHS=256

BUDGET_MAX_TIME="01:00:00:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

datamodule="conditioned_navier_stokes"
experiment="ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_f8_150m"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting AE compression 150M run"
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
