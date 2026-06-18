#!/bin/bash

set -euo pipefail
# Submit CRPS-in-ambient timing ablation for conditioned_navier_stokes only.
# Model: vit_azula_large (hidden_dim=568, n_layers=12, num_heads=8,
# patch_size=4, n_noise_channels=1024), n_members=8, AlphaFairCRPSLoss.
# Ablation path: identity encoder/decoder + processor include_global_cond=true
# (conditioning via global_cond/AdaLN, not spatial concatenation).
# See local_hydra/local_experiment/epd/conditioned_navier_stokes/
# crps_vit_azula_large_identity_global_cond.yaml for authoritative settings.
# Runs 5 epochs to estimate per-epoch wall-clock, then derive a 24h schedule:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24

declare -A EXPERIMENTS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large_identity_global_cond"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing"

for datamodule in "${!EXPERIMENTS[@]}"; do
    experiment="${EXPERIMENTS[$datamodule]}"

    echo "Submitting CRPS ambient identity+global_cond timing run"
    echo "  datamodule: ${datamodule}"
    echo "  local_experiment: ${experiment}"
    echo "  timing epochs: ${NUM_TIMING_EPOCHS}"
    echo "  budget: ${BUDGET_HOURS}h"
    echo "  run_group: ${RUN_GROUP}"
    echo ""

    uv run autocast time-epochs --kind epd --mode slurm \
        --run-group "${RUN_GROUP}" \
        --run-id "crps_ambient_identity_global_cond_b32_${datamodule}" \
        -n "${NUM_TIMING_EPOCHS}" \
        -b "${BUDGET_HOURS}" \
        local_experiment="${experiment}"

    echo ""
    echo "---"
    echo ""
done

echo "All timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/crps_ambient_identity_global_cond_b32_*/retrieve.sh; do bash \"\$f\"; done"