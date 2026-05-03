#!/bin/bash

set -euo pipefail
# CNS latent-channel sweep on the main-study AE — ~49M params, C=2.
# Same widths and FLOPs/epoch as the main-study AE, so COSINE_EPOCHS=512
# matches submit_ae_compression_large.sh (and the main-study schedule)
# on equal-schedule footing.
COSINE_EPOCHS=512

BUDGET_MAX_TIME="01:00:00:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")

datamodule="conditioned_navier_stokes"
experiment="ablations/ae_compression/conditioned_navier_stokes/ae_dc_large_c2"

for run_dry in "${RUN_DRY_STATES[@]}"; do
    dry_run_arg=()
    run_label="slurm"
    if [[ "${run_dry}" == "true" ]]; then
        dry_run_arg=(--dry-run)
        run_label="slurm --dry-run"
    fi

    echo "Submitting AE compression C=2 run"
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
