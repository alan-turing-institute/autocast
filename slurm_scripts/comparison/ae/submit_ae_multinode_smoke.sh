#!/bin/bash

set -euo pipefail
# Smoke-test multi-node DDP/NCCL with the comparison AE config.
#
# Defaults to one short conditioned_navier_stokes AE run on 2 nodes with
# 4 GPUs per node via distributed=ddp_4gpu_2node_slurm. Override the shell
# variables below when a longer run or a different AE dataset is useful.

declare -A EXPERIMENTS=(
    ["gray_scott"]="ae/gray_scott/ae_dc_large"
    ["gpe_laser_only_wake"]="ae/gpe_laser_wake_only/ae_dc_large"
    ["conditioned_navier_stokes"]="ae/conditioned_navier_stokes/ae_dc_large"
    ["advection_diffusion"]="ae/advection_diffusion/ae_dc_large"
)

DATAMODULE="${DATAMODULE:-conditioned_navier_stokes}"
if [[ -z "${EXPERIMENT:-}" ]]; then
    if [[ -z "${EXPERIMENTS[${DATAMODULE}]+x}" ]]; then
        echo "Unknown DATAMODULE=${DATAMODULE}; set EXPERIMENT explicitly."
        exit 1
    fi
    EXPERIMENT="${EXPERIMENTS[${DATAMODULE}]}"
fi
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/multinode_smoke}"
RUN_ID="${RUN_ID:-ae_multinode_${DATAMODULE}}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_TIME="${MAX_TIME:-00:00:30:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-35}"
WANDB_ENABLED="${WANDB_ENABLED:-false}"
DRY_RUN="${DRY_RUN:-false}"

dry_run_arg=()
run_label="slurm"
if [[ "${DRY_RUN}" == "true" ]]; then
    dry_run_arg=(--dry-run)
    run_label="slurm --dry-run"
fi

echo "Submitting multi-node autoencoder smoke test"
echo "  mode: ${run_label}"
echo "  datamodule: ${DATAMODULE}"
echo "  local_experiment: ${EXPERIMENT}"
echo "  distributed: ddp_4gpu_2node_slurm (2 nodes, 4 GPUs per node)"
echo "  max_epochs: ${MAX_EPOCHS}"
echo "  max_time: ${MAX_TIME}"
echo "  run_group: ${RUN_GROUP}"
echo "  run_id: ${RUN_ID}"
echo ""

uv run autocast ae --mode slurm "${dry_run_arg[@]}" \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_ID}" \
    datamodule="${DATAMODULE}" \
    local_experiment="${EXPERIMENT}" \
    distributed=ddp_4gpu_2node_slurm \
    logging.wandb.enabled="${WANDB_ENABLED}" \
    trainer.max_epochs="${MAX_EPOCHS}" \
    trainer.max_time="${MAX_TIME}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}"
