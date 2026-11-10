#!/bin/bash
set -euo pipefail

# Preview only: every workflow call below has --dry-run. No submission mode.
# Run from the checkout root with the original cache paths exported.
phase="${1:-timing}"
case "${phase}" in
    timing|train) ;;
    *) echo "Usage: bash $0 [timing|train]" >&2; exit 2 ;;
esac

RUN_GROUP="${RUN_GROUP:-$(date +%Y-%m-%d)/latent_unet_${phase}}"
RUNS=(
    "AD|advection_diffusion"
    "CNS|conditioned_navier_stokes"
    "GS|gray_scott"
    "GPE|gpe_laser_wake_only"
)

# Check all inputs before printing any previews. Timing supplies its own short
# horizon through time-epochs; it does not need a production epoch budget.
for spec in "${RUNS[@]}"; do
    IFS="|" read -r system folder <<< "${spec}"
    cache_var="AUTOCAST_UNET_${system}_CACHE"
    : "${!cache_var:?Set ${cache_var} to the original frozen AE cache}"
    if [[ "${phase}" == train ]]; then
        for method in FM DIFFUSION; do
            budget_var="AUTOCAST_UNET_${method}_${system}_EPOCHS"
            epochs="${!budget_var:-}"
            if [[ ! "${epochs}" =~ ^[1-9][0-9]*$ ]]; then
                echo "Set ${budget_var} to a measured positive integer horizon" >&2
                exit 2
            fi
        done
    fi
done

for spec in "${RUNS[@]}"; do
    IFS="|" read -r system folder <<< "${spec}"
    for method in fm diffusion; do
        run_id="latent_unet_${method}_${system}"
        overrides=(
            "local_experiment=ablations/latent_unet/${folder}/${method}_unet_matched"
            "logging.wandb.name=${run_id}"
            "hydra.launcher.gpus_per_node=4"
            "hydra.launcher.tasks_per_node=4"
            "hydra.launcher.additional_parameters.nodes=1"
        )
        if [[ "${phase}" == timing ]]; then
            uv run --frozen --no-sync autocast time-epochs \
                --kind processor --mode slurm --dry-run \
                --run-group "${RUN_GROUP}" --run-id "${run_id}" \
                -n 5 -b 24 -m 0.02 \
                "${overrides[@]}" hydra.launcher.timeout_min=240
        else
            if [[ "${method}" == fm ]]; then
                budget_var="AUTOCAST_UNET_FM_${system}_EPOCHS"
                epochs="${!budget_var}"
                quarter_epochs=$((epochs / 4))
                if ((quarter_epochs < 1)); then quarter_epochs=1; fi
                overrides+=(
                    "trainer.callbacks.0.every_n_epochs=${quarter_epochs}"
                    "trainer.callbacks.0.save_top_k=-1"
                    'trainer.callbacks.0.filename="quarter-{epoch:04d}"'
                )
            fi
            uv run --frozen --no-sync autocast processor \
                --mode slurm --dry-run \
                --run-group "${RUN_GROUP}" --run-id "${run_id}" \
                "${overrides[@]}" hydra.launcher.timeout_min=1440
        fi
    done
done
