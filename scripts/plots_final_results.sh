#!/bin/bash
set -euo pipefail

PLOTS_PATH=${PLOTS_PATH:-2026-04-27_final_plots}
RESULTS_DIR=${RESULTS_DIR:-outputs/2026-04-20_collated}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH/main_comparison_m8_complete_no_fm_amb_best_winkler}

COMMON_ARGS=(
	--dataset-order CNS
	--error-ylim 1e-5 1
	--lead-time-error-metrics vrmse rmse
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1
	--lead-time-coverage-delta
	--combined-lead-time
	--training-metrics val_loss train_loss
	--training-yscale log
	--panel-figure
	--panel-figure-no-training
	--coverage-panel-overall-rollout-windows
	--uniform-run-hue-color
	--tick-label-scale 1.5
	--axis-label-scale 1.3
	--legend-font-scale 1.5
	--short-axis-labels
	--shared-axis-labels
	--coverage-panel-height-scale 1.5
)

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

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS (ambient)" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS (ambient)" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS (ambient)" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS (ambient)" 0 eval=eval_best_multiwinkler_from0p25 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM (latent)" 3 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 3 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM (latent)" 3 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM (latent)" 3 \
	--run diff_ad64_flow_matching_vit_0f89f06_725d44a "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_gpe64_flow_matching_vit_0f89f06_3b3604d "FM (ambient)" 1 \
	--run diff_gs64_flow_matching_vit_0f89f06_6e3a299 "FM (ambient)" 1 \
	--run crps_cns64_vit_azula_large_9c98db0_4b2a1a5 "CRPS (latent)" 2 eval=eval_best_multiwinkler_from0p25 \
	--run diff_cns64_diffusion_vit_9c98db0_e9bc460 "DM (latent)" 4 eval=eval_ambient \
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
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_ambient_fm_latent_crps_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_9c98db0_2fa67c5 "ViT" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_unet_azula_large_9c98db0_65f8f71 "U-Net" 2 eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_vit_unet_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_9c98db0_6a91c49 "CRPS" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_d2a0496 "fCRPS" 1 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_2fa67c5 "afCRPS" 3 eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_crps_variants"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_9c98db0_957ff88 "m=4" 1 eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "m=8" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "m=8" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "m=8" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "m=8" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_bed4611_69c99bf "m=16" 3 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_5758ebc "m=16" 3 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_6b78265 "m=16" 3 eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_4d04729 "m=16" 3 eval=eval_best_multiwinkler_from0p25 \
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
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_ensemble_size"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_9c98db0_2fa67c5 "noise=1024" 0 eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_86e355d "noise=256" 1 eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_noise_channels"
