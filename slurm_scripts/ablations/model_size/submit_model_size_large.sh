#!/bin/bash

set -euo pipefail
# 24h production runs for the CNS model-size ablation.
#
# Populate COSINE_EPOCHS_BY_VARIANT after running
# submit_model_size_timing.sh and extracting each timing.ckpt with:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24

declare -A DATASETS=(
    ["conditioned_navier_stokes"]="conditioned_navier_stokes"
)
VARIANTS=("crps_2x" "fm_2x")

declare -A COSINE_EPOCHS_BY_VARIANT=(
    # ["conditioned_navier_stokes:crps_2x"]=...
    # ["conditioned_navier_stokes:fm_2x"]=...
)

BUDGET_MAX_TIME="00:23:59:00"
TIMEOUT_MIN=1439
RUN_DRY_STATES=("true" "false")
# Group runs under outputs/<date>/model_size/ so run dirs keep the full
# auto-naming convention (crps_<dataset>_<proc>_<hidden>_<git>_<uuid>) one
# level below the top-level analysis OUTPUTS_DIR. The ablation knob is
# surfaced via logging.wandb.name, matching ensemble_size.
RUN_GROUP="$(date +%Y-%m-%d)/model_size"

resolve_variant() {
    local variant="$1"
    experiment=""
    variant_overrides=()

    case "${variant}" in
        crps_2x)
            experiment="epd/conditioned_navier_stokes/crps_vit_azula_large"
            variant_overrides=(
                "model.processor.hidden_dim=768"
                "model.processor.n_layers=16"
                "model.processor.num_heads=8"
                "model.processor.n_noise_channels=1024"
                "model.n_members=16"
                "datamodule.batch_size=16"
            )
            ;;
        fm_2x)
            experiment="epd/conditioned_navier_stokes/fm_vit_large"
            variant_overrides=(
                "model.processor.backbone.hid_channels=896"
                "model.processor.backbone.hid_blocks=16"
                "model.processor.backbone.attention_heads=8"
            )
            ;;
        *)
            echo "FATAL: unknown variant '${variant}'" >&2
            exit 1
            ;;
    esac
}

for datamodule in "${!DATASETS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        key="${datamodule}:${variant}"
        resolve_variant "${variant}"
        cosine_epochs="${COSINE_EPOCHS_BY_VARIANT[$key]:-}"

        if [[ -z "${cosine_epochs}" ]]; then
            echo "Skipping ${key}: no COSINE_EPOCHS_BY_VARIANT entry" >&2
            continue
        fi

        quarter_epochs=$((cosine_epochs / 4))
        wandb_name="model_size_${variant}"

        for run_dry in "${RUN_DRY_STATES[@]}"; do
            dry_run_arg=()
            run_label="slurm"
            if [[ "${run_dry}" == "true" ]]; then
                dry_run_arg=(--dry-run)
                run_label="slurm --dry-run"
            fi

            echo "Submitting model-size training"
            echo "  mode: ${run_label}"
            echo "  datamodule: ${datamodule}"
            echo "  variant: ${variant}"
            echo "  local_experiment: ${experiment}"
            echo "  cosine_epochs: ${cosine_epochs}"
            echo "  wandb.name: ${wandb_name}"

            uv run autocast epd --mode slurm "${dry_run_arg[@]}" \
                --run-group "${RUN_GROUP}" \
                datamodule="${datamodule}" \
                local_experiment="${experiment}" \
                "${variant_overrides[@]}" \
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
