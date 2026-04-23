#!/bin/bash

set -euo pipefail
# 24h production runs for the ensemble-size ablation.
# Mirrors slurm_scripts/comparison/epd/submit_crps_large.sh but sweeps
# (regime, n_members, bs_per_gpu) combos from COMBOS. The active submit list
# is currently just the three non-CNS `eff_bs1024` runs; other combos are
# kept commented for later.
#
# Current coverage:
#   active now: gray_scott / gpe_laser_only_wake / advection_diffusion at eff_bs1024
#   parked for later: conditioned_navier_stokes and fixed_bs32 combos
#
# COSINE_EPOCHS_BY_COMBO can be pre-populated from submit_ensemble_timing.sh via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02
# If a key is missing, the script falls back to the matching timing.ckpt under
# outputs/*/<run_id>/timing.ckpt and derives trainer.max_epochs automatically.

# Optional per-combo cosine_epochs cache from prior timing runs (2026-04-20/21) via:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02
declare -A COSINE_EPOCHS_BY_COMBO=(
    ["conditioned_navier_stokes:fixed_bs32:16"]=253   # 334.6 s/ep
    ["conditioned_navier_stokes:eff_bs1024:16"]=249   # 340.0 s/ep
    ["gray_scott:eff_bs1024:16"]=204                  # 414.1 s/ep
    ["gpe_laser_only_wake:eff_bs1024:16"]=244         # 345.8 s/ep
    ["advection_diffusion:eff_bs1024:16"]=244         # 346.6 s/ep
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
NUM_GPUS=4
RUN_DRY_STATES=("true" "false")
# Group runs under outputs/<date>/ensemble_size/ so run dirs keep the full
# auto-naming convention (crps_<dataset>_<proc>_<git>_<uuid>) one level below
# the top-level analysis OUTPUTS_DIR. The ablation knob (regime, m) is
# surfaced via logging.wandb.name.
RUN_GROUP="$(date +%Y-%m-%d)/ensemble_size"

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

find_timing_checkpoint() {
    local run_id="$1"

    if [[ ! -d outputs ]]; then
        return 0
    fi

    find outputs -path "*/${run_id}/timing.ckpt" | sort | tail -n 1
}

derive_cosine_epochs_from_timing() {
    local timing_ckpt="$1"
    local result

    result="$(
        uv run autocast time-epochs --from-checkpoint "${timing_ckpt}" -b 24 -m 0.02
    )"

    sed -n 's/.*trainer.max_epochs=\([0-9][0-9]*\).*/\1/p' <<< "${result}" | tail -n 1
}

resolve_cosine_epochs() {
    local datamodule="$1" regime="$2" n_members="$3"
    local key="${datamodule}:${regime}:${n_members}"
    local cached="${COSINE_EPOCHS_BY_COMBO[$key]:-}"

    if [[ -n "${cached}" ]]; then
        printf '%s\n' "${cached}"
        return 0
    fi

    local run_id="crps_${datamodule}_${regime}_m${n_members}"
    local timing_ckpt
    timing_ckpt="$(find_timing_checkpoint "${run_id}")"

    if [[ -z "${timing_ckpt}" ]]; then
        return 1
    fi

    derive_cosine_epochs_from_timing "${timing_ckpt}"
}

declare -A DATASETS=(
    ["gray_scott"]="epd/gray_scott/crps_vit_azula_large"
    ["gpe_laser_only_wake"]="epd/gpe_laser_wake_only/crps_vit_azula_large"
    # ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
    ["advection_diffusion"]="epd/advection_diffusion/crps_vit_azula_large"
)

declare -A REGIMES_BY_DATASET=(
    ["gray_scott"]="eff_bs1024"
    ["gpe_laser_only_wake"]="eff_bs1024"
    # ["conditioned_navier_stokes"]="fixed_bs32 eff_bs1024"
    ["advection_diffusion"]="eff_bs1024"
)

# (regime, n_members, bs_per_gpu) triples, matching submit_ensemble_timing.sh.
COMBOS=(
    # "fixed_bs32 16 32"
    "eff_bs1024 16 16"
)

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

        key="${datamodule}:${regime}:${n_members}"
        if ! cosine_epochs="$(resolve_cosine_epochs "${datamodule}" "${regime}" "${n_members}")"; then
            echo "Skipping ${key}: no timing-derived cosine_epochs available" >&2
            continue
        fi
        if [[ -z "${cosine_epochs}" ]]; then
            echo "Skipping ${key}: could not parse trainer.max_epochs from timing output" >&2
            continue
        fi

        quarter_epochs=$((cosine_epochs / 4))
        wandb_name="ensemble_m${n_members}_${regime}"

        for run_dry in "${RUN_DRY_STATES[@]}"; do
            dry_run_arg=()
            run_label="slurm"
            if [[ "${run_dry}" == "true" ]]; then
                dry_run_arg=(--dry-run)
                run_label="slurm --dry-run"
            fi

            echo "Submitting ensemble-size training"
            echo "  mode: ${run_label}"
            echo "  datamodule: ${datamodule}"
            echo "  regime: ${regime}  n_members: ${n_members}  bs_per_gpu: ${bs_per_gpu}"
            echo "  cosine_epochs: ${cosine_epochs}"
            echo "  wandb.name: ${wandb_name}"

            uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
                --run-group "${RUN_GROUP}" \
                datamodule="${datamodule}" \
                local_experiment="${experiment}" \
                model.n_members="${n_members}" \
                datamodule.batch_size="${bs_per_gpu}" \
                logging.wandb.enabled=true \
                logging.wandb.name="${wandb_name}" \
                optimizer.cosine_epochs="${cosine_epochs}" \
                hydra.launcher.timeout_min="${TIMEOUT_MIN}" \
                trainer.max_time="${BUDGET_MAX_TIME}" \
                +trainer.max_epochs="${cosine_epochs}" \
                trainer.callbacks.0.every_n_epochs="${quarter_epochs}" \
                trainer.callbacks.0.save_top_k=-1 \
                trainer.callbacks.0.filename=\"quarter-{epoch:04d}\"
        done
    done
done
