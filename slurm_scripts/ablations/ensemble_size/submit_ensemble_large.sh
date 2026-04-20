#!/bin/bash

set -euo pipefail
# 24h production runs for the ensemble-size ablation (CNS only for now).
# Mirrors slurm_scripts/comparison/epd/submit_crps_large.sh but sweeps
# (regime, n_members, bs_per_gpu) combos from COMBOS. Current defaults are
# m=16 pilot runs, but the table is designed to be extended.
#
# COSINE_EPOCHS_BY_COMBO is populated from submit_ensemble_timing.sh via
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24
# and then pasted in below.

# Per-combo cosine_epochs from timing runs (2026-04-20) via:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24 -m 0.02
declare -A COSINE_EPOCHS_BY_COMBO=(
    ["conditioned_navier_stokes:fixed_bs32:16"]=253   # 334.6 s/ep
    ["conditioned_navier_stokes:eff_bs1024:16"]=249   # 340.0 s/ep
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

declare -A DATASETS=(
    ["conditioned_navier_stokes"]="epd/conditioned_navier_stokes/crps_vit_azula_large"
    # ["gray_scott"]="epd/gray_scott/crps_vit_azula_large"
)

# (regime, n_members, bs_per_gpu) triples, matching submit_ensemble_timing.sh.
COMBOS=(
    "fixed_bs32 16 32"
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
        key="${datamodule}:${regime}:${n_members}"
        cosine_epochs="${COSINE_EPOCHS_BY_COMBO[$key]:-}"

        if [[ -z "${cosine_epochs}" ]]; then
            echo "Skipping ${key}: no COSINE_EPOCHS_BY_COMBO entry" >&2
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
