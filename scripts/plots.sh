#!/bin/bash

# CRPS variants
autocast-plots --results-dir outputs/2026-04-01_collated \
	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (large)" 0 \
	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (large)" 1 \
	--run crps_sw2d64_vit_azula_large_60114bf_5d5c8e1 "ViT (1024)" 1 \
	--run crps_sw2d64_vit_azula_large_concat_60114bf_acf9189 "ViT (concat)" 1 \
	--run epd_sw2d64_vit_azula_large_8c5f696_c5f3ea9 "ViT (MAE loss)" 1 \
	--run crps_sw2d64_vit_azula_large_feb5e43_c630eda "ViT (afCRPS loss)" 1 \
	--output-dir outputs/2026-04-01_collated/plots/sw_crps_variants

# # 23hrs
# autocast-plots --results-dir outputs/2026-04-01_collated --sort Date --filter "Dataset=SW2D64 AND (Model=AzulaViTProcessor OR Model=FlowMatchingProcessor) AND Resolution=64x64 AND scale=large" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (large)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (large)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_90e3465 "ViT (23hrs)" \
# 	--output-dir outputs/2026-04-01_collated/plots/sw_23hrs

# # Comparison of 1:4, 1:1 and 2:1
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM (1:4)" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (1:4)" \
# 	--run crps_cns64_vit_azula_large_8fe25aa_d105b90 "ViT (1:4)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (1:4)" \
# 	--run diff_cached_latents_flow_matching_vit_640_feb5e43_83aa8bf "FM (1:1)" \
# 	--run diff_cached_latents_flow_matching_vit_640_feb5e43_fe5c2fa "FM (1:1)" \
# 	--run crps_cns64_vit_azula_large_feb5e43_91c22c0 "ViT (1:1)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_bb3bf07 "ViT (1:1)" \
# 	--run diff_cached_latents_flow_matching_vit_640_feb5e43_1de6fc1 "FM (2:1)" \
# 	--run diff_cached_latents_flow_matching_vit_640_feb5e43_0f2b8e5 "FM (2:1)" \
# 	--run crps_cns64_vit_azula_large_feb5e43_b8f68ac "ViT (2:1)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_ffb68ce "ViT (2:1)" \
# 	--color-by-label \
# 	--output-dir outputs/2026-04-01_collated/plots/1_4_vs_1_1_vs_2_1

# # Comparison of 1:4 and 8x gradient steps
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM (1:4)" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (1:4)" \
# 	--run diff_cns64_flow_matching_vit_c3c9713_d2bcaa2 "FM (64, 24hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_c3c9713_e4e3ba5 "FM (64, 24hrs)" \
# 	--color-by-label \
# 	--output-dir outputs/2026-04-01_collated/plots/fm_8x

# # Comparison with 128x128
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_shallow_water2d_128_vit_azula_large_05cf9ff_29b49a6 "ViT 64x64" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT 128x128" \
# 	--color-by-label \
# 	--output-dir outputs/2026-04-01_collated/plots/128x128

# # Comparison with ODE steps
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_c3c9713_d2bcaa2_25 "FM (25 steps)" \
# 	--run diff_cns64_flow_matching_vit_c3c9713_d2bcaa2 "FM (50 steps)" \
# 	--run diff_cns64_flow_matching_vit_c3c9713_d2bcaa2_100 "FM (100 steps)" \
# 	--run diff_sw2d64_flow_matching_vit_c3c9713_e4e3ba5_25 "FM (25 steps)" \
# 	--run diff_sw2d64_flow_matching_vit_c3c9713_e4e3ba5 "FM (50 steps)" \
# 	--run diff_sw2d64_flow_matching_vit_c3c9713_e4e3ba5_100 "FM (100 steps)" \
# 	--color-by-label \
# 	--group-hues 0 0 0 0 0 0 \
# 	--output-dir outputs/2026-04-01_collated/plots/ode_steps
