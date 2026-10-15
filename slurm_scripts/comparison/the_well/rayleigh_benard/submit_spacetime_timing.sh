#!/bin/bash
# Profile the new RB configs; this script never submits full training runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_common.sh"
rb_require_cache

RUN_GROUP="${RUN_GROUP:-timing/rb_spacetime_masked_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_BASE="${OUTPUT_BASE:-${RB_REPO_ROOT}/outputs}"
# Two validation cycles distinguish startup from recurring epoch overhead.
TIMING_EPOCHS="${TIMING_EPOCHS:-16}"
if (( TIMING_EPOCHS < 8 || TIMING_EPOCHS % 8 != 0 )); then
    echo "TIMING_EPOCHS must be a multiple of eight to include validation." >&2
    exit 1
fi
RUN_KEYS=("$@")
if (($# == 0)); then RUN_KEYS=(fm crps); fi

for method in "${RUN_KEYS[@]}"; do
    case "${method}" in
        fm|crps) ;;
        *) echo "Unknown method: ${method} (expected fm or crps)" >&2; exit 1 ;;
    esac
    while IFS= read -r run_dry; do
        rb_submit time-epochs "${run_dry}" \
            --kind processor --output-base "${OUTPUT_BASE}" \
            --run-group "${RUN_GROUP}" --run-id "${method}" \
            -n "${TIMING_EPOCHS}" -b 24 -m 0.05 \
            "local_experiment=the_well/rayleigh_benard/${method}_spacetime_latent" \
            "datamodule.data_path=${RB_CACHE_DIR}" \
            "logging.wandb.enabled=true" "output.save_config=true" \
            "logging.wandb.name=rb_spacetime_masked_${method}_timing" \
            "logging.wandb.group=rb_spacetime_masked_timing" \
            "hydra.launcher.nodes=1" \
            "hydra.launcher.timeout_min=${TIMEOUT_MIN:-20}" \
            "hydra.launcher.cpus_per_task=16" \
            "trainer.callbacks.0.every_n_epochs=${TIMING_EPOCHS}"
    done < <(rb_run_dry_states)
done
