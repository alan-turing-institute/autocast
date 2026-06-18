#!/bin/bash

set -euo pipefail
# Timing runs for the CNS model-size ablation.
#
# Baselines come from slurm_scripts/comparison/README.md and the CNS
# local_experiment presets under local_hydra/local_experiment/epd/.
# This sweep adds an aspect-preserving 3-point scan of even-depth ambient
# variants (heads fixed at 8, width chosen per baseline width/depth aspect
# ratio):
#   - CRPS ViT (baseline 568/12/8 ~80.8M, aspect 47.3):
#       0p4x  376/8/8   ~31.6M   0.391x
#       2x    768/16/8  ~169.3M  2.094x
#   - FM ViT   (baseline 704/12/8 ~80.0M, aspect 58.7):
#       0p4x  472/8/8   ~25.6M   0.319x
#       2x    896/16/8  ~168.3M  2.103x
# CRPS variants all set n_members=16, batch_size=16 to keep the effective
# per-GPU minibatch at 256 (matching baseline n_members=8, batch_size=32).
#
# Runs 5 epochs to produce timing.ckpt, then extract cosine_epochs via:
#   uv run autocast time-epochs --from-checkpoint <path>/timing.ckpt -b 24

# Use the comparison-study baseline presets plus CLI overrides so the
# auto-generated run directories stay descriptive, matching the ensemble-size
# ablation pattern.
declare -A DATASETS=(
    ["conditioned_navier_stokes"]="conditioned_navier_stokes"
)
VARIANTS=(
    "crps_0p4x"
    "crps_2x"
    "fm_0p4x"
    "fm_2x"
)

BUDGET_HOURS=24
NUM_TIMING_EPOCHS=5
RUN_GROUP="$(date +%Y-%m-%d)/timing_model_size"

resolve_variant() {
    local variant="$1"
    experiment=""
    variant_overrides=()

    case "${variant}" in
        crps_0p4x)
            experiment="epd/conditioned_navier_stokes/crps_vit_azula_large"
            variant_overrides=(
                "model.processor.hidden_dim=376"
                "model.processor.n_layers=8"
                "model.processor.num_heads=8"
                "model.processor.n_noise_channels=1024"
                "model.n_members=16"
                "datamodule.batch_size=16"
            )
            ;;
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
        fm_0p4x)
            experiment="epd/conditioned_navier_stokes/fm_vit_large"
            variant_overrides=(
                "model.processor.backbone.hid_channels=472"
                "model.processor.backbone.hid_blocks=8"
                "model.processor.backbone.attention_heads=8"
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
        resolve_variant "${variant}"
        run_id="model_size_${variant}_${datamodule}"

        echo "Submitting model-size timing run"
        echo "  datamodule: ${datamodule}"
        echo "  variant: ${variant}"
        echo "  local_experiment: ${experiment}"
        echo "  run_id: ${run_id}"
        echo ""

        uv run autocast time-epochs --kind epd --mode slurm \
            --run-group "${RUN_GROUP}" \
            --run-id "${run_id}" \
            -n "${NUM_TIMING_EPOCHS}" \
            -b "${BUDGET_HOURS}" \
            datamodule="${datamodule}" \
            local_experiment="${experiment}" \
            "${variant_overrides[@]}"

        echo ""
        echo "---"
        echo ""
    done
done

echo "All model-size timing jobs submitted."
echo ""
echo "Once SLURM jobs complete, collect all results with:"
echo "  for f in outputs/${RUN_GROUP}/model_size_*/retrieve.sh; do bash \"\$f\"; done"
