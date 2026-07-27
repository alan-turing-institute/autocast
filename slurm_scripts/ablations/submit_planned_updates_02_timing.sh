#!/bin/bash

set -euo pipefail

# Sequential five-epoch timing runs for planned updates batch 02.
#
# This batch times the MC-dropout MSE + L2 baseline with p=0.1 across all four
# comparison datasets. Run the jobs sequentially in Isambard's interactive
# reservation, using the same lightweight checkpoint stack as the original
# April main-comparison timing runs. Do not use the full default callback stack
# here: its 5%-progress cadence would compress roughly 20 production snapshots
# into each five-epoch timing window.
#
# Corrected results from the 2026-07-27 interactive runs (24h budget,
# 2% margin, April callback stack):
#   gray_scott:                  33.7 s/epoch -> 2513 epochs
#   gpe_laser_only_wake:         27.4 s/epoch -> 3087 epochs
#   conditioned_navier_stokes:   30.6 s/epoch -> 2769 epochs
#   advection_diffusion:         26.8 s/epoch -> 3154 epochs
# These epoch counts are pinned in submit_planned_updates_02_large.sh.

BUDGET_HOURS=24
NUM_TIMING_EPOCHS="${NUM_TIMING_EPOCHS:-5}"
MARGIN=0.02
RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/timing_planned_updates_02}"
RESERVATION="${RESERVATION:-interactive}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/../.."
    pwd
)"
cd "${PROJECT_ROOT}"

declare -A EXPERIMENTS=(
    ["gray_scott"]="ablations/mc_dropout/gray_scott/mse_vit_azula_mc_dropout_large"
    ["gpe_laser_only_wake"]="ablations/mc_dropout/gpe_laser_wake_only/mse_vit_azula_mc_dropout_large"
    ["conditioned_navier_stokes"]="ablations/mc_dropout/conditioned_navier_stokes/mse_vit_azula_mc_dropout_large"
    ["advection_diffusion"]="ablations/mc_dropout/advection_diffusion/mse_vit_azula_mc_dropout_large"
)

DATASETS=(
    "gray_scott"
    "gpe_laser_only_wake"
    "conditioned_navier_stokes"
    "advection_diffusion"
)

for datamodule in "${DATASETS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"
    run_id="mc_dropout_mse_${datamodule}"
    run_dir="outputs/${RUN_GROUP}/${run_id}"
    mkdir -p "${run_dir}"

    echo "Running planned updates batch 02 timing job"
    echo "  run_id: ${run_id}"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  margin: ${MARGIN}"
    echo "  run_group: ${RUN_GROUP}"
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
        --job-name="mc_mse_timing_${datamodule}" \
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

    echo
    echo ---
    echo
done

echo "All planned updates batch 02 timing jobs completed."
echo
echo "Recompute any recommendation with:"
echo "  uv run --frozen autocast time-epochs \\"
echo "    --from-checkpoint outputs/<date>/timing_planned_updates_02/<run_id>/timing.ckpt \\"
echo "    -b ${BUDGET_HOURS} -m ${MARGIN}"
