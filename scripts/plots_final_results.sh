#!/bin/bash
set -euo pipefail

PLOTS_PATH=${PLOTS_PATH:-2026-04-27_final_plots}
RESULTS_DIR=${RESULTS_DIR:-outputs/2026-04-20_collated}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH/main_comparison_m8_complete_no_fm_amb_best_winkler}

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS" 0 eval=eval_best_multiwinkler_from0p25 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM" 3 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM" 3 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM" 3 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM" 3 \
	--dataset-order AD CNS GS GPE \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
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
