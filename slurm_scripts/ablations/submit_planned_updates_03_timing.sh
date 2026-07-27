#!/bin/bash

set -euo pipefail

# Five-epoch timing runs for planned updates batch 03.
#
# This batch adds the parameter-matched CRPS FNO architecture ablation across
# all four main-comparison datasets. Run the jobs sequentially in Isambard's
# interactive reservation, using the same lightweight checkpoint stack as the
# original April main-comparison timing runs. Do not use the full default
# callback stack here: its 5%-progress cadence would compress roughly 20
# production snapshots into this five-epoch timing window.
#
# Each resulting timing.ckpt is consumed by
# submit_planned_updates_03_large.sh to derive the number of epochs that fit
# the same 24h budget as the ViT baseline.
#
# Results from the 2026-07-27 interactive runs (24h budget, 2% margin):
#   gray_scott:                 144.9 s/epoch -> 584 epochs
#   gpe_laser_only_wake:        121.8 s/epoch -> 695 epochs
#   conditioned_navier_stokes:  123.0 s/epoch -> 688 epochs
#   advection_diffusion:        118.7 s/epoch -> 713 epochs
# These epoch counts are pinned in submit_planned_updates_03_large.sh.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
MARGIN=0.02
RUN_GROUP="$(date +%Y-%m-%d)/timing_planned_updates_03"
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

DATASETS=(
    "gray_scott"
    "gpe_laser_only_wake"
    "conditioned_navier_stokes"
    "advection_diffusion"
)

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
        trainer=fm_main_comparison \
        local_experiment="${experiment}"

    echo ""
    echo "---"
    echo ""
done

echo "All planned updates batch 03 timing jobs completed."
echo ""
echo "Recompute any recommendation with:"
echo "  uv run --frozen autocast time-epochs \\"
echo "    --from-checkpoint outputs/<date>/timing_planned_updates_03/<run_id>/timing.ckpt \\"
echo "    -b ${BUDGET_HOURS} -m ${MARGIN}"
