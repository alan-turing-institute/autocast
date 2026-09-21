#!/bin/bash
set -euo pipefail

# Run from the repository root. Existing historical links are left in place.
RESULTS_DIR=${RESULTS_DIR:-outputs/2026-07-24_collated}
EXTENSION_DIR=../2026-09-18/diffusion_unet_extensions
RUNS=(
	crps_ad64_unet_azula_large_138dd8c_79a7707
	crps_gs64_unet_azula_large_138dd8c_03eb2bf
	crps_gpe64_unet_azula_large_138dd8c_79c0d48
	diff_ad64_diffusion_vit_d80b5b3_eceb84e
	diff_gs64_diffusion_vit_d80b5b3_9476ca0
	diff_gpe64_diffusion_vit_d80b5b3_4856c88
)

# Validate the entire set before creating any links.
for run in "${RUNS[@]}"; do
	target="$EXTENSION_DIR/$run"
	if [[ ! -d "$RESULTS_DIR/$target" ]]; then
		echo "Missing run: $RESULTS_DIR/$target" >&2
		exit 1
	fi
	if [[ -e "$RESULTS_DIR/$run" || -L "$RESULTS_DIR/$run" ]]; then
		if [[ ! -L "$RESULTS_DIR/$run" || "$(readlink "$RESULTS_DIR/$run")" != "$target" ]]; then
			echo "Refusing to replace existing path: $RESULTS_DIR/$run" >&2
			exit 1
		fi
	fi
done
for run in "${RUNS[@]}"; do
	if [[ ! -L "$RESULTS_DIR/$run" ]]; then
		ln -s "$EXTENSION_DIR/$run" "$RESULTS_DIR/$run"
	fi
done
echo "Six extension run links are ready in $RESULTS_DIR."
