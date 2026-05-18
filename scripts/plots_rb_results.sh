#!/bin/bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-outputs/2026-05-18_rb}
PLOTS_PATH=${PLOTS_PATH:-2026-05-18_plots_rb}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH/rb_pair}
FIGURE_FORMATS=${FIGURE_FORMATS:-png}

FIGURE_FORMAT_ARRAY=()
read -r -a FIGURE_FORMAT_ARRAY <<< "${FIGURE_FORMATS//,/ }"

HUE_A=0
HUE_B=1

echo "Writing plots under: $OUTPUT_DIR"

autocast-plots --results-dir "$RESULTS_DIR" \
	--figure-formats "${FIGURE_FORMAT_ARRAY[@]}" \
	--run diff_rayleigh_benard_flow_matching_vit_1de6ca4_5a7a7bb "FM (masked window)" "$HUE_A" \
	--run diff_rayleigh_benard_flow_matching_vit_65377a2_acb8513 "FM (whole window)" "$HUE_B" \
	--metrics vrmse_v2 coverage crps ssr \
	--error-yscale linear \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse_v2 \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--lead-time-coverage-delta \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--coverage-panel-overall-rollout-windows \
	--uniform-run-hue-color \
	--tick-label-scale 1.5 \
	--axis-label-scale 1.3 \
	--legend-font-scale 1.5 \
	--short-axis-labels \
	--shared-axis-labels \
	--coverage-panel-height-scale 1.5 \
	--output-dir "$OUTPUT_DIR"
