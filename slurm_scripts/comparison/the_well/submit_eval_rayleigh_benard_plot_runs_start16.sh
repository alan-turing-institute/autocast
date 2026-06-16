#!/bin/bash

set -euo pipefail
# Submit the Rayleigh-Benard eval reruns used by scripts/plots_rb_results.sh.
#
# Each child script keeps the checkpoint, eval.mode, sampler, memory, and metric
# defaults for its run family. The common convention is eval.rollout_start=16
# and output folders suffixed with start16, so these reruns do not overwrite the
# older default eval outputs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Plot refreshes only need metrics/coverage outputs, not timing benchmark CSVs.
export EVAL_BENCHMARK_ENABLED="${EVAL_BENCHMARK_ENABLED:-false}"
export EVAL_BENCHMARK_ROLLOUT_ENABLED="${EVAL_BENCHMARK_ROLLOUT_ENABLED:-false}"

RUN_SCRIPTS=(
    "${SCRIPT_DIR}/submit_eval_rayleigh_benard_fm_vit.sh"
    "${SCRIPT_DIR}/effective_batch_24h/eval/submit_eval_crps_latent_24h.sh"
    "${SCRIPT_DIR}/effective_batch_24h/eval/submit_eval_fm_latent_24h.sh"
    "${SCRIPT_DIR}/effective_batch_24h/eval/submit_eval_crps_ambient_24h.sh"
    "${SCRIPT_DIR}/effective_batch_24h/eval/submit_eval_crps_ambient_lola_pixel_24h.sh"
    "${SCRIPT_DIR}/submit_eval_rayleigh_benard_diffusion_vit_large_lola.sh"
    "${SCRIPT_DIR}/submit_eval_rayleigh_benard_diffusion_vit_large_lola_ddpm.sh"
    "${SCRIPT_DIR}/submit_eval_rayleigh_benard_diffusion_vit_large_lola_ab.sh"
)

for run_script in "${RUN_SCRIPTS[@]}"; do
    echo
    echo "==> ${run_script}"
    bash "${run_script}"
done
