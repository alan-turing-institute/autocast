#!/bin/bash

set -euo pipefail
# Timing runs for the ensemble-size ablation.
#
# Two regimes (see README). Current defaults are m=16, but COMBOS is meant
# to be extended as the ablation grows:
#   fixed_bs32: fixed datamodule.batch_size=32/GPU (same as main CRPS run).
#   eff_bs1024: fixed global effective batch = 1024 (matches main budget).
#
# Each combo inherits the per-dataset CRPS-ambient baseline and overrides
# model.n_members + datamodule.batch_size via CLI — no new experiment
# configs. Timing runs disable wandb + skip_test; produce timing.ckpt.
#
# Current coverage:
#   conditioned_navier_stokes: fixed_bs32 + eff_bs1024
#   gray_scott / gpe_laser_only_wake / advection_diffusion: eff_bs1024 only

declare -A DATASETS=(
    ["gray_scott"]="epd/gray_scott/crps_vit_azula_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large"
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
    ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large"
)

declare -A REGIMES_BY_DATASET=(
    ["gray_scott"]="eff_bs1024"
    ["gpe_laser_only_wake"]="eff_bs1024"
    ["conditioned_navier_stokes"]="fixed_bs32 eff_bs1024"
    ["advection_diffusion"]="eff_bs1024"
)

# (regime, n_members, bs_per_gpu) triples. bs_per_gpu chosen so that:
#   fixed_bs32:    bs_per_gpu                         = 32
#   eff_bs1024:    bs_per_gpu × n_members × 4 GPUs   = 1024
COMBOS=(
    "fixed_bs32 16 32"
    "eff_bs1024 16 16"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
NUM_GPUS=4
RUN_GROUP="$(date +%Y-%m-%d)/timing_ensemble_m16"

# Sanity-check the COMBOS table up front: bail before submitting anything
# if a (regime, n_members, bs_per_gpu) triple violates its regime invariant.
assert_combo() {
    local regime="$1" n_members="$2" bs_per_gpu="$3"
    case "${regime}" in
        fixed_bs32)
            if (( bs_per_gpu != 32 )); then
                echo "FATAL: fixed_bs32 combo expects bs_per_gpu=32, got bs_per_gpu=${bs_per_gpu}" >&2
                exit 1
            fi
            ;;
        eff_bs1024)
            local expected=$((bs_per_gpu * n_members * NUM_GPUS))
            if (( expected != 1024 )); then
                echo "FATAL: eff_bs1024 combo (m=${n_members}, bs=${bs_per_gpu}) gives effective global ${expected}, expected 1024" >&2
                exit 1
            fi
            ;;
        *)
            echo "FATAL: unknown regime '${regime}'" >&2
            exit 1
            ;;
    esac
}

dataset_supports_regime() {
    local datamodule="$1" regime="$2"
    local supported="${REGIMES_BY_DATASET[$datamodule]:-}"
    local supported_regime

    for supported_regime in ${supported}; do
        if [[ "${supported_regime}" == "${regime}" ]]; then
            return 0
        fi
    done

    return 1
}

for combo in "${COMBOS[@]}"; do
    read -r regime n_members bs_per_gpu <<< "${combo}"
    assert_combo "${regime}" "${n_members}" "${bs_per_gpu}"
done

for datamodule in "${!DATASETS[@]}"; do
    experiment="${DATASETS[$datamodule]}"

    for combo in "${COMBOS[@]}"; do
        read -r regime n_members bs_per_gpu <<< "${combo}"
        if ! dataset_supports_regime "${datamodule}" "${regime}"; then
            echo "Skipping ${datamodule}/${regime}: regime not enabled for this dataset" >&2
            continue
        fi

        run_id="crps_${datamodule}_${regime}_m${n_members}"

        echo "Submitting ensemble-size timing run"
        echo "  datamodule: ${datamodule}"
        echo "  regime: ${regime}"
        echo "  n_members: ${n_members}"
        echo "  bs_per_gpu: ${bs_per_gpu}"
        echo "  run_id: ${run_id}"
        echo ""

        uv run autocast time-epochs --kind epd --mode slurm \
            --run-group "${RUN_GROUP}" \
            --run-id "${run_id}" \
            -n "${NUM_TIMING_EPOCHS}" \
            -b "${BUDGET_HOURS}" \
            local_experiment="${experiment}" \
            model.n_members="${n_members}" \
            datamodule.batch_size="${bs_per_gpu}"

        echo ""
        echo "---"
        echo ""
    done
done

echo "All ensemble-size timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/*/retrieve.sh; do bash \"\$f\"; done"
