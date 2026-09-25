#!/bin/bash

set -euo pipefail

# Production-callback timing runs for planned updates batch 03.
#
# This batch adds the parameter-matched CRPS FNO architecture ablation across
# all four main-comparison datasets. Run the jobs sequentially in Isambard's
# interactive reservation with the production validation checkpoint stack.
# The plotting callback is no longer part of the default trainer. The
# 5%-progress snapshot callback is explicitly suppressed because resolving its
# cadence against a short timing run would compress roughly 20 production
# snapshots into five epochs. Best-val, MultiCoverage, MultiWinkler, and EMA
# remain active.
#
# Each resulting timing.ckpt is consumed by
# submit_planned_updates_03_large.sh to derive the number of epochs that fit
# the same 24h budget as the ViT baseline.
#
# Reviewed production-callback results from the 2026-07-27 interactive runs
# (24h budget, 2% margin):
#   gray_scott:                 143.2 s/epoch -> 591 epochs
#   gpe_laser_only_wake:        123.8 s/epoch -> 683 epochs
#   conditioned_navier_stokes:  126.7 s/epoch -> 668 epochs
#   advection_diffusion:        119.5 s/epoch -> 708 epochs
# These epoch counts are pinned in submit_planned_updates_03_large.sh.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
MARGIN=0.02
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing_planned_updates_03_production_callbacks_5ep}"
RESERVATION="${RESERVATION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)"
cd "${PROJECT_ROOT}"

declare -A EXPERIMENTS=(
    ["gray_scott"]="ablations/arch_unet_fno_vit/gray_scott/crps_fno_80m"
    ["gpe_laser_only_wake"]="ablations/arch_unet_fno_vit/gpe_laser_wake_only/crps_fno_80m"
    ["conditioned_navier_stokes"]="ablations/arch_unet_fno_vit/conditioned_navier_stokes/crps_fno_80m"
    ["advection_diffusion"]="ablations/arch_unet_fno_vit/advection_diffusion/crps_fno_80m"
)

if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
else
    DATASETS=(
        "gray_scott"
        "gpe_laser_only_wake"
        "conditioned_navier_stokes"
        "advection_diffusion"
    )
fi

for datamodule in "${DATASETS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="fno_m8_crps_${datamodule}"
    run_dir="outputs/${RUN_GROUP}/${run_id}"
    mkdir -p "${run_dir}"

    echo "Running planned updates batch 03 timing job"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  margin: ${MARGIN}"
    echo "  run_group: ${RUN_GROUP}"
    echo "  run_id: ${run_id}"
    echo "  reservation: ${RESERVATION}"

    srun \
        --reservation="${RESERVATION}" \
        --nodes=1 \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --mem=0 \
        --exclusive \
        --time="${TIME_LIMIT}" \
        --kill-on-bad-exit=1 \
        --job-name="fno_timing_${datamodule}" \
        --output="${run_dir}/slurm-%j.out" \
        --error="${run_dir}/slurm-%j.err" \
        uv run --frozen autocast time-epochs \
        --kind epd \
        --mode local \
        --workdir "${run_dir}" \
        --run-group "${RUN_GROUP}" \
        --run-id "${run_id}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        -m "${MARGIN}" \
        trainer=default \
        local_experiment="${experiment}" \
        trainer.callbacks.0.every_n_train_steps_fraction=null \
        +trainer.callbacks.0.every_n_train_steps=999999999 \
        +trainer.callbacks.0.every_n_epochs=0 \
        trainer.callbacks.0.save_last=false

    echo ""
    echo "---"
    echo ""
done

echo "All planned updates batch 03 timing jobs completed."
echo ""
echo "Recompute any recommendation with:"
echo "  uv run --frozen autocast time-epochs \\"
echo "    --from-checkpoint outputs/<date>/timing_planned_updates_03_production_callbacks_5ep/<run_id>/timing.ckpt \\"
echo "    -b ${BUDGET_HOURS} -m ${MARGIN}"
