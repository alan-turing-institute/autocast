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
MAX_EPOCHS="${MAX_EPOCHS:-10}"
MAX_TIME="${MAX_TIME:-00:01:00:00}"
TIMEOUT_MIN="${TIMEOUT_MIN:-70}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
LOG_GPU_UTIL="${LOG_GPU_UTIL:-true}"
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
echo "  datamodule.num_workers: ${NUM_WORKERS}"
echo "  cpus_per_task: ${CPUS_PER_TASK}"
echo "  log_gpu_util: ${LOG_GPU_UTIL} (per-rank GPU util in the SLURM log)"
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
    +log_gpu_util="${LOG_GPU_UTIL}" \
    +trainer.max_epochs="${MAX_EPOCHS}" \
    trainer.max_time="${MAX_TIME}" \
    datamodule.num_workers="${NUM_WORKERS}" \
    hydra.launcher.cpus_per_task="${CPUS_PER_TASK}" \
    hydra.launcher.timeout_min="${TIMEOUT_MIN}"

echo ""
echo "After the job finishes, summarise per-GPU utilization (one line per epoch per rank) with:"
echo "  grep GpuUtilizationLogCallback <output_dir>/slurm-<jobid>.out | sort"
echo "Expect MAX_EPOCHS x 8 lines (epoch=N, global_rank=0..7 across 2 hosts);"
echo "look for high busy(>=80%%) and near-zero idle(<10%%) on every rank/epoch."
