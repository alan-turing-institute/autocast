#!/bin/bash
set -euo pipefail

PLOTS_PATH=${PLOTS_PATH:-2026-05-19_final_plots}
RESULTS_DIR=${RESULTS_DIR:-outputs/2026-05-15_collated}
OUTPUT_DIR=${OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH/main_comparison_m8_complete_no_fm_amb_best_winkler}
FIGURE_FORMATS=${FIGURE_FORMATS:-png}
PAPER_OUTPUT_DIR=${PAPER_OUTPUT_DIR:-$RESULTS_DIR/$PLOTS_PATH/paper_figures}
PAPER_USE_TEX=${PAPER_USE_TEX:-false}
PAPER_PANEL_LABELS=${PAPER_PANEL_LABELS:-true}

# Keep colours stable across all final plots. These are indices into matplotlib's
# tab10 palette as used by the explicit hue argument.
HUE_CRPS=0
HUE_FM_LATENT=1
HUE_CRPS_LATENT=2
HUE_FM_AMBIENT=3
HUE_DM=4
HUE_ABLATION_ALT_1=1
HUE_ABLATION_ALT_2=2
HUE_ABLATION_ALT_3=3
HUE_ODE_ABLATION_STEPS=5
HUE_ODE_MAIN=$HUE_FM_LATENT

PAPER_MAIN_FIGURES=false
FOUR_DS_ABLATION=false
ONE_DS_ABLATION=false
PAPER_ONLY=false

usage() {
	cat <<'EOF'
Usage: scripts/plots_final_results.sh [OPTIONS]

Options:
  --paper-main-figures  Add paper-width main-result figures to OUTPUT_DIR,
                        including the multi-metric lead-time summary.
  --four-ds-ablation    Add four-dataset ablation figures to multi-dataset ablations.
  --one-ds-ablation     Add the one-dataset ablation paper figure.
  --paper-figures       Add all optional paper-width figures above.
  --paper-only          Only run/copy paper figures, writing PNG and PDF.
  --paper-output-dir DIR
                        Copy all paper_*.png/pdf figures into DIR.
                        Default: <results-dir>/<plots-path>/paper_figures.
  --paper-use-tex       Render text/math with external LaTeX.
  --no-paper-panel-labels
                        Disable a), b), c) labels on combined paper figures.
  --pdf                 Write both PNG and PDF figures.
  --figure-formats FMT  Write selected formats, comma-separated or quoted.
                        Example: --figure-formats png,pdf
  -h, --help            Show this help.

By default the script preserves the standard output set and does not write
paper_*.png files. Pass --paper-figures to generate the paper-ready variants.
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--paper-main-figures)
			PAPER_MAIN_FIGURES=true
			;;
		--four-ds-ablation)
			FOUR_DS_ABLATION=true
			;;
		--one-ds-ablation)
			ONE_DS_ABLATION=true
			;;
		--paper-figures)
			PAPER_MAIN_FIGURES=true
			FOUR_DS_ABLATION=true
			ONE_DS_ABLATION=true
			;;
		--paper-only)
			PAPER_MAIN_FIGURES=true
			FOUR_DS_ABLATION=true
			ONE_DS_ABLATION=true
			PAPER_ONLY=true
			;;
		--paper-output-dir)
			if [[ $# -lt 2 ]]; then
				echo "--paper-output-dir requires a value" >&2
				exit 2
			fi
			PAPER_OUTPUT_DIR=$2
			shift
			;;
		--paper-use-tex)
			PAPER_USE_TEX=true
			;;
		--no-paper-panel-labels)
			PAPER_PANEL_LABELS=false
			;;
		--pdf)
			FIGURE_FORMATS="png pdf"
			;;
		--figure-formats)
			if [[ $# -lt 2 ]]; then
				echo "--figure-formats requires a value" >&2
				exit 2
			fi
			FIGURE_FORMATS=${2//,/ }
			shift
			;;
		-h | --help)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 2
			;;
	esac
	shift
done

FIGURE_FORMAT_ARRAY=()
read -r -a FIGURE_FORMAT_ARRAY <<< "$FIGURE_FORMATS"
if [[ "$PAPER_MAIN_FIGURES" == true || "$FOUR_DS_ABLATION" == true || "$ONE_DS_ABLATION" == true ]]; then
	case " ${FIGURE_FORMAT_ARRAY[*]} " in
		*" png "*) ;;
		*) FIGURE_FORMAT_ARRAY+=(png) ;;
	esac
	case " ${FIGURE_FORMAT_ARRAY[*]} " in
		*" pdf "*) ;;
		*) FIGURE_FORMAT_ARRAY+=(pdf) ;;
	esac
fi
FIGURE_FORMAT_ARGS=(--figure-formats "${FIGURE_FORMAT_ARRAY[@]}")
BASE_PLOT_ARGS=("${FIGURE_FORMAT_ARGS[@]}")
if [[ "$PAPER_USE_TEX" == true ]]; then
	BASE_PLOT_ARGS+=(--paper-use-tex)
fi
if [[ "$PAPER_ONLY" == true ]]; then
	BASE_PLOT_ARGS+=(--paper-only)
fi
if [[ "$PAPER_PANEL_LABELS" != true ]]; then
	BASE_PLOT_ARGS+=(--no-paper-panel-labels)
fi


COMMON_ARGS=(
	"${BASE_PLOT_ARGS[@]}"
	--dataset-order CNS
	--error-ylim 1e-5 1
	--lead-time-error-metrics vrmse
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

if [[ "$ONE_DS_ABLATION" == true ]]; then
	COMMON_ARGS+=(--one-ds-ablation)
fi

PAPER_MAIN_ARG=
if [[ "$PAPER_MAIN_FIGURES" == true ]]; then
	PAPER_MAIN_ARG=--paper-main-figures
fi

FOUR_DS_ABLATION_ARG=
if [[ "$FOUR_DS_ABLATION" == true ]]; then
	FOUR_DS_ABLATION_ARG=--four-ds-ablation
fi

ALL_DATASET_COMMON_ARGS=(
	"${BASE_PLOT_ARGS[@]}"
	--dataset-order AD CNS GS GPE
	--error-ylim 1e-5 1
	--lead-time-error-metrics vrmse
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1
	--lead-time-coverage-delta
	--combined-lead-time
	--training-metrics val_loss train_loss
	--training-yscale log
	--panel-figure
	--panel-figure-no-training
	--coverage-panel-overall-rollout-windows
	--tick-label-scale 1.5
	--axis-label-scale 1.3
	--legend-font-scale 1.5
	--short-axis-labels
	--shared-axis-labels
	--coverage-panel-height-scale 1.5
)

if [[ "$FOUR_DS_ABLATION" == true ]]; then
	ALL_DATASET_COMMON_ARGS+=(--four-ds-ablation)
fi

echo "Writing plots under: $RESULTS_DIR/$PLOTS_PATH"
echo "Writing figure formats: ${FIGURE_FORMAT_ARRAY[*]}"
if [[ "$PAPER_MAIN_FIGURES" == true || "$FOUR_DS_ABLATION" == true || "$ONE_DS_ABLATION" == true ]]; then
	echo "Collecting paper figures under: $PAPER_OUTPUT_DIR"
fi
if [[ "$PAPER_USE_TEX" == true ]]; then
	echo "Rendering plot text with LaTeX."
fi
if [[ "$PAPER_ONLY" == true ]]; then
	echo "Paper-only mode enabled."
fi
if [[ "$PAPER_MAIN_FIGURES" != true && "$FOUR_DS_ABLATION" != true && "$ONE_DS_ABLATION" != true ]]; then
	echo "Paper-ready outputs are disabled. Run with --paper-figures to write paper_*.png files."
fi

autocast-plots --results-dir "$RESULTS_DIR" \
	"${BASE_PLOT_ARGS[@]}" \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM" "$HUE_FM_LATENT" \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM" "$HUE_FM_LATENT" \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM" "$HUE_FM_LATENT" \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM" "$HUE_FM_LATENT" \
	--dataset-order AD CNS GS GPE \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse \
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
	${PAPER_MAIN_ARG:+$PAPER_MAIN_ARG} \
	--output-dir "$OUTPUT_DIR"

if [[ "$PAPER_ONLY" != true ]]; then
	# Same as above but rendered smaller so the two coverage panels read well when
	# paired side-by-side in LaTeX (each at ~0.5\textwidth).
	autocast-plots --results-dir "$RESULTS_DIR" \
		"${BASE_PLOT_ARGS[@]}" \
		--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
		--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
		--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
		--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
		--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM" "$HUE_FM_LATENT" \
		--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM" "$HUE_FM_LATENT" \
		--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM" "$HUE_FM_LATENT" \
		--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM" "$HUE_FM_LATENT" \
		--dataset-order AD CNS GS GPE \
		--error-ylim 1e-5 1 \
		--lead-time-error-metrics vrmse \
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
		--figure-scale 0.6 \
		--output-dir "${OUTPUT_DIR}_pairfig"
fi

autocast-plots --results-dir "$RESULTS_DIR" \
	"${ALL_DATASET_COMMON_ARGS[@]}" \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS (ambient)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS (ambient)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS (ambient)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS (ambient)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_3b47441_3ad3562 "CRPS (latent)" "$HUE_CRPS_LATENT" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_4b2a1a5 "CRPS (latent)" "$HUE_CRPS_LATENT" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_3b47441_1c8e446 "CRPS (latent)" "$HUE_CRPS_LATENT" eval=eval_best_multiwinkler_from0p25 \
	--run diff_ad64_flow_matching_vit_0f89f06_725d44a "FM (ambient)" "$HUE_FM_AMBIENT" \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" "$HUE_FM_AMBIENT" \
	--run diff_gpe64_flow_matching_vit_0f89f06_3b3604d "FM (ambient)" "$HUE_FM_AMBIENT" \
	--run diff_gs64_flow_matching_vit_0f89f06_6e3a299 "FM (ambient)" "$HUE_FM_AMBIENT" \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM (latent)" "$HUE_FM_LATENT" \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" "$HUE_FM_LATENT" \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM (latent)" "$HUE_FM_LATENT" \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM (latent)" "$HUE_FM_LATENT" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_ambient_fm_latent_crps_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" "$HUE_FM_LATENT" \
	--run diff_cns64_diffusion_vit_0c75022_80967c4 "DM (latent)" "$HUE_DM" eval=eval_encode_once \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_dm_latent"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=1" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode001 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=1" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode001 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=1" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode001 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=1" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode001 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=5" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode005 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=5" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode005 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=5" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode005 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=5" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode005 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=10" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode010 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=10" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode010 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=10" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode010 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=10" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode010 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=25" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode025 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=25" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode025 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=25" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode025 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=25" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode025 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=50" "$HUE_ODE_MAIN" eval=eval \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=50" "$HUE_ODE_MAIN" eval=eval \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=50" "$HUE_ODE_MAIN" eval=eval \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=50" "$HUE_ODE_MAIN" eval=eval \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "ODE=100" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode100 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "ODE=100" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode100 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "ODE=100" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode100 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "ODE=100" "$HUE_ODE_ABLATION_STEPS" eval=eval_encode_once_ode100 \
	"${ALL_DATASET_COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_fm_ode_steps"

autocast-plots --results-dir "$RESULTS_DIR" \
	"${BASE_PLOT_ARGS[@]}" \
	--run crps_ad64_vit_azula_large_0f89f06_4667606 "CRPS (best val loss)" "$HUE_ABLATION_ALT_2" eval=eval \
	--run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (best val loss)" "$HUE_ABLATION_ALT_2" eval=eval \
	--run crps_gpe64_vit_azula_large_0f89f06_d337bd8 "CRPS (best val loss)" "$HUE_ABLATION_ALT_2" eval=eval \
	--run crps_gs64_vit_azula_large_0f89f06_779325a "CRPS (best val loss)" "$HUE_ABLATION_ALT_2" eval=eval \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS (multiwinkler)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS (multiwinkler)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS (multiwinkler)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS (multiwinkler)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--dataset-order AD CNS GS GPE \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse \
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
	${FOUR_DS_ABLATION_ARG:+$FOUR_DS_ABLATION_ARG} \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_crps_multiwinkler_eval_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (6x)" "$HUE_FM_LATENT" \
	--run diff_cns64_flow_matching_vit_43cbdde_bb55197 "FM (24x)" "$HUE_ABLATION_ALT_2" eval=eval_encode_once \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_fm_autoencoder"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS (channel concatenation)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_2fa67c5 "CRPS (backbone modulation)" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_global_conditioning_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "ViT" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_unet_azula_large_9c98db0_65f8f71 "U-Net" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_vit_unet_m8"

autocast-plots --results-dir "$RESULTS_DIR" \
	"${ALL_DATASET_COMMON_ARGS[@]}" \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "CRPS (\$\alpha\$fCRPS loss)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "CRPS (\$\alpha\$fCRPS loss)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "CRPS (\$\alpha\$fCRPS loss)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "CRPS (\$\alpha\$fCRPS loss)" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_6a91c49 "CRPS (CRPS loss)" "$HUE_ABLATION_ALT_1" eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_5a8c216_9978d9b "CRPS (fCRPS loss)" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_d2a0496 "CRPS (fCRPS loss)" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_5a8c216_2b1460a "CRPS (fCRPS loss)" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_5a8c216_2ced703 "CRPS (fCRPS loss)" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_crps_variants"

autocast-plots --results-dir "$RESULTS_DIR" \
	"${BASE_PLOT_ARGS[@]}" \
	--run crps_cns64_vit_azula_large_9c98db0_957ff88 "M=4" "$HUE_ABLATION_ALT_1" eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_189c141_bbb0bc8 "M=4" "$HUE_ABLATION_ALT_1" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_3b47441_4944cce "M=4" "$HUE_ABLATION_ALT_1" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_189c141_ce8db86 "M=4" "$HUE_ABLATION_ALT_1" eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_bed4611_da01a04 "M=8" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "M=8" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_e0a6df5 "M=8" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_828a161 "M=8" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_ad64_vit_azula_large_bed4611_69c99bf "M=16" "$HUE_ABLATION_ALT_3" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_bed4611_5758ebc "M=16" "$HUE_ABLATION_ALT_3" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gpe64_vit_azula_large_bed4611_6b78265 "M=16" "$HUE_ABLATION_ALT_3" eval=eval_best_multiwinkler_from0p25 \
	--run crps_gs64_vit_azula_large_bed4611_4d04729 "M=16" "$HUE_ABLATION_ALT_3" eval=eval_best_multiwinkler_from0p25 \
	--dataset-order AD CNS GS GPE \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse \
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
	${FOUR_DS_ABLATION_ARG:+$FOUR_DS_ABLATION_ARG} \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_ensemble_size"

autocast-plots --results-dir "$RESULTS_DIR" \
	--run crps_cns64_vit_azula_large_bed4611_c99f534 "noise=1024, h=568" "$HUE_CRPS" eval=eval_best_multiwinkler_from0p25 \
	--run crps_cns64_vit_azula_large_9c98db0_86e355d "noise=256, h=704" "$HUE_ABLATION_ALT_2" eval=eval_best_multiwinkler_from0p25 \
	"${COMMON_ARGS[@]}" \
	--output-dir "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_noise_channels"

if [[ "$PAPER_MAIN_FIGURES" == true || "$FOUR_DS_ABLATION" == true || "$ONE_DS_ABLATION" == true ]]; then
	mkdir -p "$PAPER_OUTPUT_DIR/png" "$PAPER_OUTPUT_DIR/pdf" "$PAPER_OUTPUT_DIR/tables"
	SKIPPED_PAPER_DIRS=(
		ablation_cns_dm
		ablation_fm_ema
	)
	for ext in png pdf; do
		rm -f "$PAPER_OUTPUT_DIR/$ext/"*_paper_one_ds_ablation_b."$ext"
		for skipped_dir in "${SKIPPED_PAPER_DIRS[@]}"; do
			rm -f "$PAPER_OUTPUT_DIR/$ext/${skipped_dir}_"*."$ext"
		done
	done
	rm -f "$PAPER_OUTPUT_DIR/tables/"*.csv "$PAPER_OUTPUT_DIR/tables/"*.tex
	copied=0
	while IFS= read -r fig; do
		src_dir=$(basename "$(dirname "$fig")")
		fig_name=$(basename "$fig")
		case "$fig_name" in
			paper_one_ds_ablation_b.*)
				continue
				;;
		esac
		case "$src_dir" in
			ablation_cns_dm | ablation_fm_ema)
				continue
				;;
		esac
		ext=${fig##*.}
		dest_name="${src_dir}_${fig_name}"
		cp "$fig" "$PAPER_OUTPUT_DIR/$ext/$dest_name"
		copied=$((copied + 1))
	done < <(
		find "$RESULTS_DIR/$PLOTS_PATH" \
			-path "$PAPER_OUTPUT_DIR" -prune -o \
			-path "$RESULTS_DIR/$PLOTS_PATH/ablation_fm_ema" -prune -o \
			-path "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_dm" -prune -o \
			-type f \( -name 'paper_*.png' -o -name 'paper_*.pdf' \) \
			-print
	)
	echo "Copied $copied paper figure files to: $PAPER_OUTPUT_DIR/{png,pdf}"
	copied_tables=0
	while IFS= read -r table; do
		src_dir=$(basename "$(dirname "$table")")
		case "$src_dir" in
			ablation_cns_dm | ablation_fm_ema)
				continue
				;;
		esac
		cp "$table" "$PAPER_OUTPUT_DIR/tables/${src_dir}_$(basename "$table")"
		copied_tables=$((copied_tables + 1))
	done < <(
		find "$RESULTS_DIR/$PLOTS_PATH" \
			-path "$PAPER_OUTPUT_DIR" -prune -o \
			-path "$RESULTS_DIR/$PLOTS_PATH/ablation_fm_ema" -prune -o \
			-path "$RESULTS_DIR/$PLOTS_PATH/ablation_cns_dm" -prune -o \
			-type f \( -name 'single_step_overall_results.csv' -o -name 'single_step_overall_results.tex' \) \
			-print
	)
	echo "Copied $copied_tables table result files to: $PAPER_OUTPUT_DIR/tables"
fi
