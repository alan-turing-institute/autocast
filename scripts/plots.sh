#!/bin/bash

# # 23hrs
# autocast-plots --results-dir outputs/2026-04-01_collated --sort Date --filter "Dataset=SW2D64 AND (Model=AzulaViTProcessor OR Model=FlowMatchingProcessor) AND Resolution=64x64 AND scale=large" \
# 	--runs \
# 		diff_sw2d64_flow_matching_vit_4d5fcbd_067529b \
# 		crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 \
# 		crps_sw2d64_vit_azula_large_feb5e43_90e3465 \
# 	--labels \
# 		"FM (large)" \
# 		"ViT (large)" \
# 		"ViT (23hrs)" \
# 	--output-dir outputs/2026-04-01_collated/plots/sw_23hrs

# Comparison of 1:4, 1:1 and 2:1
autocast-plots --results-dir outputs/2026-04-01_collated \
	--runs \
		diff_cns64_flow_matching_vit_55a4d1a_511e7b1 \
		diff_sw2d64_flow_matching_vit_4d5fcbd_067529b \
		crps_cns64_vit_azula_large_8fe25aa_d105b90 \
		crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 \
		diff_cached_latents_flow_matching_vit_640_feb5e43_83aa8bf \
		diff_cached_latents_flow_matching_vit_640_feb5e43_fe5c2fa \
		crps_cns64_vit_azula_large_feb5e43_91c22c0 \
		crps_sw2d64_vit_azula_large_feb5e43_bb3bf07 \
		diff_cached_latents_flow_matching_vit_640_feb5e43_1de6fc1 \
		diff_cached_latents_flow_matching_vit_640_feb5e43_0f2b8e5 \
		crps_cns64_vit_azula_large_feb5e43_b8f68ac \
		crps_sw2d64_vit_azula_large_feb5e43_ffb68ce \
	--color-by-label \
	--labels \
		"FM (1:4)" \
		"FM (1:4)" \
		"ViT (1:4)" \
		"ViT (1:4)" \
		"FM (1:1)" \
		"FM (1:1)" \
		"ViT (1:1)" \
		"ViT (1:1)" \
		"FM (2:1)" \
		"FM (2:1)" \
		"ViT (2:1)" \
		"ViT (2:1)" \
	--output-dir outputs/2026-04-01_collated/plots/1_4_vs_1_1_vs_2_1
