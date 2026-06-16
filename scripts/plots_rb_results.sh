#!/bin/bash
set -euo pipefail

RESULTS_DIR=${RESULTS_DIR:-outputs/2026-05-18_rb}
PLOTS_PATH=${PLOTS_PATH:-2026-05-18_plots_rb}
OUTPUT_ROOT=${OUTPUT_ROOT:-${OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH}}
FIGURE_FORMATS=${FIGURE_FORMATS:-png}
PLOT_CMD=${PLOT_CMD:-"uv run autocast-plots"}

FIGURE_FORMAT_ARRAY=()
read -r -a FIGURE_FORMAT_ARRAY <<< "${FIGURE_FORMATS//,/ }"
PLOT_CMD_ARRAY=()
read -r -a PLOT_CMD_ARRAY <<< "$PLOT_CMD"

HUE_CRPS_AMBIENT=0
HUE_FM_LATENT=1
HUE_CRPS_LATENT=2
HUE_CRPS_LOLA_AMBIENT=3
HUE_DIFFUSION=4
HUE_FM_MASKED=5
HUE_FM_NON_MASKED=6

RB_FM_MASKED=diff_rayleigh_benard_flow_matching_vit_1de6ca4_5a7a7bb
RB_FM_NON_MASKED=diff_rayleigh_benard_flow_matching_vit_65377a2_acb8513
RB_FM_LATENT_24H=the_well_rayleigh_benard_effbatch24h_fm_latent_b256
RB_CRPS_LATENT_24H=the_well_rayleigh_benard_effbatch24h_crps_latent_b32_m8
RB_CRPS_AMBIENT_24H=the_well_rayleigh_benard_effbatch24h_crps_ambient_b32_m8
RB_CRPS_LOLA_AMBIENT_24H=the_well_rayleigh_benard_effbatch_crps_ambient_lola_pixel_b32_m8_24hr
RB_DIFFUSION_4096=the_well_rayleigh_benard_diffusion_vit_large_lola4096

COMMON_ARGS=(
	--results-dir "$RESULTS_DIR"
	--figure-formats "${FIGURE_FORMAT_ARRAY[@]}"
	--metrics vrmse_v2 coverage crps ssr \
	--error-yscale linear \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse_v2 psrmse_low psrmse_mid psrmse_high \
	--lead-time-coverage-metrics spread_skill_lola \
	--lead-time-coverage-delta \
	--combined-lead-time \
	--paper-rb-mosaic \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--coverage-panel-overall-rollout-windows \
	--tick-label-scale 1.5 \
	--axis-label-scale 1.3 \
	--legend-font-scale 1.5 \
	--short-axis-labels \
	--shared-axis-labels \
	--coverage-panel-height-scale 1.5
)

plot_group() {
	local name=$1
	shift

	local output_dir="$OUTPUT_ROOT/$name"
	echo "Writing $name plots under: $output_dir"
	"${PLOT_CMD_ARRAY[@]}" "${COMMON_ARGS[@]}" "$@" --output-dir "$output_dir"
}

# Config check:
# - 4096 runs use max_epochs=4096, max_time=00:23:59:00, batch_size=64,
#   devices=4, so their effective batch size is 256.
# - 24h comparison runs use max_time=00:23:59:00 with run-specific
#   max_epochs. FM latent has effective batch size 1024; the CRPS runs have
#   effective batch size 128.

plot_group masked_vs_non_masked \
	--run "$RB_FM_MASKED" "FM masked (4096, eff_bs=256)" "$HUE_FM_MASKED" \
	--run "$RB_FM_NON_MASKED" "FM non-masked (4096, eff_bs=256)" "$HUE_FM_NON_MASKED" \
	--uniform-run-hue-color

plot_group 24hrs \
	--run "$RB_CRPS_LATENT_24H" "CRPS latent (24h, eff_bs=128)" "$HUE_CRPS_LATENT" \
	--run "$RB_FM_LATENT_24H" "FM latent (24h, eff_bs=1024)" "$HUE_FM_LATENT" \
	--run "$RB_CRPS_AMBIENT_24H" "CRPS ambient (24h, eff_bs=128)" "$HUE_CRPS_AMBIENT" \
	--run "$RB_CRPS_LOLA_AMBIENT_24H" "CRPS LoLA ambient (24h, eff_bs=128)" "$HUE_CRPS_LOLA_AMBIENT" \
	--uniform-run-hue-color

plot_group 4096 \
	--run "$RB_FM_NON_MASKED" "FM non-masked (4096, eff_bs=256)" "$HUE_FM_NON_MASKED" \
	--run "$RB_FM_MASKED" "FM masked (4096, eff_bs=256)" "$HUE_FM_MASKED" \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256)" "$HUE_DIFFUSION" eval=eval_encode_once \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256, DDPM)" "$HUE_DIFFUSION" eval=eval_encode_once_ddpm50 \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256, ABo3)" "$HUE_DIFFUSION" eval=eval_encode_once_ab16_o3_relative
	# --run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256, AB)" "$HUE_DIFFUSION" eval=eval_encode_once_ab16 \

plot_group 4096_start16 \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256)" "$HUE_DIFFUSION" eval=eval_encode_once_start16 \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256, DDPM)" "$HUE_DIFFUSION" eval=eval_encode_once_ddpm50_start16 \
	--run "$RB_DIFFUSION_4096" "Diffusion LoLA (4096, eff_bs=256, ABo3)" "$HUE_DIFFUSION" eval=eval_encode_once_ab16_o3_start16
	# --run "$RB_FM_NON_MASKED" "FM non-masked (4096, eff_bs=256)" "$HUE_FM_NON_MASKED" \
	# --run "$RB_FM_MASKED" "FM masked (4096, eff_bs=256)" "$HUE_FM_MASKED" \
