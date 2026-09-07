#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset_root="${AUTOCAST_DATASETS:-/data/aiphys/scratch/swe-crps-issue-153/datasets}"
run_root="${SWE_CRPS_RUN_ROOT:-/data/aiphys/scratch/swe-crps-issue-153/runs}"

stochastic_data="${dataset_root}/swe_crps_32_128f_spinup40/stats.yml"
deterministic_data="${dataset_root}/swe_crps_deterministic_32_128f_spinup5/stats.yml"

if [[ ! -f "${stochastic_data}" || ! -f "${deterministic_data}" ]]; then
    echo "SWE CRPS datasets not found under ${dataset_root}" >&2
    exit 1
fi

export AUTOCAST_DATASETS="${dataset_root}"
mkdir -p "${run_root}"
cd "${project_root}"

run_experiment() {
    local run_name="$1"
    local experiment="$2"
    local datamodule="$3"
    local experiment_name="$4"
    local run_dir="${run_root}/${run_name}"

    if [[ -e "${run_dir}" ]]; then
        echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
        exit 1
    fi

    echo "[$(date --iso-8601=seconds)] Starting ${run_name}"
    uv run train_encoder_processor_decoder \
        "experiment=${experiment}" \
        "datamodule=${datamodule}" \
        "experiment_name=${experiment_name}" \
        seed=42 \
        logging.wandb.enabled=false \
        output.skip_test=true \
        "hydra.run.dir=${run_dir}"

    if [[ ! -f "${run_dir}/encoder_processor_decoder.ckpt" ]]; then
        echo "Run finished without a final checkpoint: ${run_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${run_dir}/lightning_logs/version_0/checkpoints/snapshot-step-00008192.ckpt" ]]; then
        echo "Run finished without its step-8192 snapshot: ${run_dir}" >&2
        exit 1
    fi
    echo "[$(date --iso-8601=seconds)] Completed ${run_name}"
}

run_experiment \
    "azula-vit-concat-residual-stochastic-long-8192" \
    "epd_crps_vit_concat_residual_32_long" \
    "shallow_water2d_crps_32" \
    "swe_crps_azula_vit_concat_stochastic_long_8192"

run_experiment \
    "azula-vit-adaln-residual-stochastic-long-8192" \
    "epd_crps_vit_adaln_residual_32_long" \
    "shallow_water2d_crps_32" \
    "swe_crps_azula_vit_adaln_stochastic_long_8192"

run_experiment \
    "azula-vit-concat-residual-deterministic-long-8192" \
    "epd_crps_vit_concat_residual_32_long" \
    "shallow_water2d_crps_deterministic_32" \
    "swe_crps_azula_vit_concat_deterministic_long_8192"

run_experiment \
    "azula-vit-adaln-residual-deterministic-long-8192" \
    "epd_crps_vit_adaln_residual_32_long" \
    "shallow_water2d_crps_deterministic_32" \
    "swe_crps_azula_vit_adaln_deterministic_long_8192"
