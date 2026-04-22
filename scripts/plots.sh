#!/bin/bash

PLOTS_PATH=2026-04-20_plots_prelim_m16


# Comparison with previous CNS results
# m=8 
# --run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (24hrs, update, n=1024, afCRPS, h=568, layers=12)" 0 \
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run crps_cns64_vit_azula_large_1ed9013_ab31602 "CRPS (24hrs, prelim, n=1024, afCRPS, h=632, layers=10)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (24hrs, update, n=1024, afCRPS, h=568, layers=12, m=16)" 0 \
	--run diff_cns64_flow_matching_vit_1ed9013_893624d "FM (24hrs, prelim)" 2 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--dataset-order CNS \
	--error-ylim 1e-3 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/cns_comparison_with_exploratory_results

# Main comparison: CRPS ambient variants, CRPS processor-on-latents, FM ambient (EPD)
# --run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient)" 0 \
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run crps_ad64_vit_azula_large_0f89f06_4667606 "CRPS (ambient)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (ambient m=16)" 0 \
	--run crps_gpe64_vit_azula_large_0f89f06_d337bd8 "CRPS (ambient)" 0 \
	--run crps_gs64_vit_azula_large_0f89f06_779325a "CRPS (ambient) " 0 \
	--run diff_ad64_flow_matching_vit_0f89f06_725d44a "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_gpe64_flow_matching_vit_0f89f06_3b3604d "FM (ambient)" 1 \
	--run diff_gs64_flow_matching_vit_0f89f06_6e3a299 "FM (ambient)" 1 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM (latent)" 2 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM (latent)" 2 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM (latent)" 2 \
	--dataset-order AD CNS GS GPE \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/main_comparison


# Main comparison (with m=16 runs around 2/3 complete): CRPS ambient variants, CRPS processor-on-latents, FM ambient (EPD)
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run crps_ad64_vit_azula_large_ac1bb06_ef6368d "CRPS (ambient, m=16, 16hrs)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (ambient, m=16)" 0 \
	--run crps_gpe64_vit_azula_large_ac1bb06_638585e "CRPS (ambient, m=16, 16hrs)" 0 \
	--run crps_gs64_vit_azula_large_ac1bb06_639963f "CRPS (ambient, m=16, 16hrs) " 0 \
	--run diff_ad64_flow_matching_vit_0f89f06_725d44a "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_gpe64_flow_matching_vit_0f89f06_3b3604d "FM (ambient)" 1 \
	--run diff_gs64_flow_matching_vit_0f89f06_6e3a299 "FM (ambient)" 1 \
	--run diff_ad64_flow_matching_vit_09490da_dae1382 "FM (latent)" 2 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run diff_gpe64_flow_matching_vit_09490da_47bf39a "FM (latent)" 2 \
	--run diff_gs64_flow_matching_vit_09490da_7e9e331 "FM (latent)" 2 \
	--dataset-order AD CNS GS GPE\
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/main_comparison_m16_part_complete

# Ablation with CRPS ViT using global cond instead of permute_concat and channels last
# --run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient)" 0 \
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient, m=8)" 0 \
	--run crps_cns64_vit_azula_large_0f89f06_cf53b48 "CRPS (ambient, identity+global-cond, m=8)" 0 \
	--dataset-order CNS \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/ablation_crps_vit_global_cond


# Usign cached latents for evaluation instead of AE-ambient latents for CRPS latent variant
# --run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient)" 0 \
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient, m=8)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (ambient m=16)" 3 \
	--run crps_cns64_vit_azula_large_0f89f06_e7e60d9 "CRPS (latent, CRPS in ambient, m=8)" 0 \
	--run crps_cns64_vit_azula_large_09490da_8b7573d "CRPS (latent, CRPS in latent, m=8)" 0 \
	--dataset-order CNS \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/ablation_crps_vit_latent


# Ablation with CRPS ensemble size
# TODO: update with final runs
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run crps_cns64_vit_azula_large_0f89f06_5b7332b "CRPS (ambient, m=8)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_5e157a5 "CRPS (ambient, m=16, bs=32)" 0 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (ambient, m=16, eff_bs=1024)" 0 \
	--dataset-order CNS \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/ablation_crps_ensemble_size_complete

# Model size ablation
autocast-plots --results-dir outputs/2026-04-20_collated \
	--run diff_cns64_flow_matching_vit_0f89f06_483bb70 "FM (ambient)" 1 \
	--run diff_cns64_flow_matching_vit_896_3a69487_e894c55 "FM (ambient, 2x, 16hrs)" 1 \
	--run diff_cns64_flow_matching_vit_09490da_636fcc3 "FM (latent)" 2 \
	--run crps_cns64_vit_azula_large_0db40e1_dcd79e4 "CRPS (ambient, m=16)" 0 \
	--run crps_cns64_vit_azula_large_768_3a69487_1d7da5f "CRPS (ambient, m=16, 2x, 16hrs)" 0 \
	--dataset-order CNS \
	--error-ylim 1e-5 1 \
	--lead-time-error-metrics vrmse rmse \
	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
	--combined-lead-time \
	--training-metrics val_loss train_loss \
	--training-yscale log \
	--panel-figure \
	--panel-figure-no-training \
	--output-dir outputs/2026-04-20_collated/$PLOTS_PATH/ablation_crps_model_size

# ---

# PLOTS_PATH=2026-04-08_plots

# # CRPS variants
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (large)"\
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (large)"\
# 	--run crps_sw2d64_vit_azula_large_60114bf_5d5c8e1 "ViT (1024)"\
# 	--run crps_sw2d64_vit_azula_large_concat_60114bf_acf9189 "ViT (concat)"\
# 	--run epd_sw2d64_vit_azula_large_8c5f696_c5f3ea9 "ViT (MAE loss)"\
# 	--run crps_sw2d64_vit_azula_large_feb5e43_c630eda "ViT (afCRPS loss)"\
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/sw_crps_variants

# # 23hrs
# autocast-plots --results-dir outputs/2026-04-01_collated --sort Date --filter "Dataset=SW2D64 AND (Model=AzulaViTProcessor OR Model=FlowMatchingProcessor) AND Resolution=64x64 AND scale=large" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (large)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (large)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_90e3465 "ViT (23hrs)" \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--color-by-label \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/sw_23hrs

# # Comparison of 1:4, 1:1 and 2:1
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM (1:4)" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (1:4)" \
# 	--run crps_cns64_vit_azula_large_8fe25aa_d105b90 "ViT (1:4)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (1:4)" \
# 	--run diff_cns64_flow_matching_vit_13eba33_1f075ac "FM (1:1)" \
# 	--run diff_sw2d64_flow_matching_vit_13eba33_0bc663c "FM (1:1)" \
# 	--run crps_cns64_vit_azula_large_feb5e43_91c22c0 "ViT (1:1)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_bb3bf07 "ViT (1:1)" \
# 	--run diff_cns64_flow_matching_vit_13eba33_9305e6a "FM (2:1)" \
# 	--run diff_sw2d64_flow_matching_vit_13eba33_02cb549 "FM (2:1)" \
# 	--run crps_cns64_vit_azula_large_feb5e43_b8f68ac "ViT (2:1)" \
# 	--run crps_sw2d64_vit_azula_large_feb5e43_ffb68ce "ViT (2:1)" \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/1_4_vs_1_1_vs_2_1_no_cached_latents

# # Comparison of 1:4 and 8x gradient steps
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM (1:4)" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (1:4)" \
# 	--run diff_cns64_flow_matching_vit_c3c9713_d2bcaa2 "FM (64, 24hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_c3c9713_e4e3ba5 "FM (64, 24hrs)" \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/fm_8x

# # Comparison with 128x128
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_shallow_water2d_128_vit_azula_large_05cf9ff_29b49a6 "ViT 128x128" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT 64x64" \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/128x128

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
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/ode_steps


# # Comparison of 1:4, longer 1:4, smoother optimizer 1:4
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_cns64_vit_azula_large_8fe25aa_d105b90 "ViT" \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM" \
# 	--run diff_cns64_flow_matching_vit_13eba33_82f423d "FM (36hrs)" \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM (lr=1e-4)" \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/longer_training_and_smoother_optimizer

# # Compare SW, CNS, GPE
# # TODO: update SW, CNS and GPE with same training schedule (GPE currently 2e-4)
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_cns64_vit_azula_large_eb4a4f6_294dc78 "ViT" \
# 	--run crps_sw2d64_vit_azula_large_eb4a4f6_682f33b "ViT" \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM" \
# 	--run crps_gpe_laser_only_wake_vit_azula_large_feb5e43_699855c "ViT" \
# 	--run diff_gpe_laser_only_wake_flow_matching_vit_2007857_7ee7dea "FM" \
# 	--color-by-label \
# 	--dataset-order SW CNS GPE \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/sw_cns_gpe

# # Compare DM, FM, ViT for SW, CNS
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_cns64_vit_azula_large_eb4a4f6_294dc78 "ViT" \
# 	--run crps_sw2d64_vit_azula_large_eb4a4f6_682f33b "ViT" \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM" \
# 	--run diff_cns64_diffusion_vit_fc5b63a_3c040a7 "DM" \
# 	--run diff_sw2d64_diffusion_vit_3dd3cf5_e849015 "DM" \
# 	--color-by-label \
# 	--dataset-order SW CNS \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_fm_dm

# # Compare ViT and FM for batch_size 32 and 128 trained over 12hrs 1GPU / 3hrs 4GPU
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM" \
# 	--run crps_cns64_vit_azula_large_8fe25aa_d105b90 "ViT (n=256, CRPS, h=768, bs=32)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (n=256, CRPS, h=768, bs=32)" \
# 	--run crps_cns64_vit_azula_large_eb4a4f6_294dc78 "ViT (n=256, CRPS, h=768, bs=128)" \
# 	--run crps_sw2d64_vit_azula_large_eb4a4f6_682f33b "ViT (n=256, CRPS, h=768, bs=128)" \
# 	--color-by-label \
# 	--dataset-order SW CNS \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_fm_vits_bs_32_vs_128


# # Compare DM, FM, ViT for SW, CNS
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM" \
# 	--run crps_cns64_vit_azula_large_eb4a4f6_294dc78 "ViT (n=256, CRPS, h=768)" \
# 	--run crps_sw2d64_vit_azula_large_eb4a4f6_682f33b "ViT (n=256, CRPS, h=768)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_4268ef8 "ViT (n=1024, afCRPS, h=768)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_4268ef8 "ViT (n=1024, afCRPS, h=768)" \
# 	--run crps_sw2d64_vit_azula_large_468b553_0bdf23f "ViT (n=256, CRPS, h=768, bs=256, bf16)" \
# 	--color-by-label \
# 	--dataset-order SW CNS \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_fm_vits_comparison_before_24hrs


# # # Compare FM, ViT for SW, CNS with long 24hrs runs
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM (3hrs)" \
# 	--run diff_cns64_flow_matching_vit_1ed9013_893624d "FM (24hrs)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (3hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_ab31602 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM (3hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_1ed9013_5529d56 "FM (24hrs)" \
# 	--run crps_sw2d64_vit_azula_large_1ed9013_6ab9fa2 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--dataset-order SW CNS \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_fm_vits_comparison_24hrs


# # Compare FM, ViT for SW, CNS with long 24hrs runs
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM (3hrs)" \
# 	--run diff_cns64_flow_matching_vit_1ed9013_893624d "FM (24hrs)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (3hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_ab31602 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM (3hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_1ed9013_5529d56 "FM (24hrs)" \
# 	--run crps_sw2d64_vit_azula_large_1ed9013_6ab9fa2 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--dataset-order SW CNS \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_fm_vits_comparison_24hrs_v
	

# # Compare FM, ViT for CNS with varying batch size, training time, precision and model size
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM (3hrs)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_ab31602 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (3hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_6bb8774_bb210b3 "ViT (3hrs, n=1024, afCRPS, h=632, bf16)" \
# 	--run crps_cns64_vit_azula_large_52e1abc_1ca7096 "ViT (3hrs, n=4096, afCRPS, h=608, bf16, bs=32)" \
# 	--run crps_cns64_vit_azula_large_concat_c8ff6c2_e36b786 "ViT (3hrs, n=1024, afCRPS, h=632, concat, bf16)" \
# 	--dataset-order CNS \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/crps_vits_bf16_4096_vs_1024
# 	# --color-by-label \

# # Compare FM, ViT for SW, CNS with long 24hrs runs
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_1ed9013_893624d "FM (24hrs)" \
# 	--run diff_cns64_flow_matching_vit_1ed9013_893624d_ode_method "FM (24hrs, Adams-Bashforth)" \
# 	--dataset-order CNS \
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/fm_24hrs_adams_bashforth
	

# # Revisit recents runs including comparison:
# # - between optimizer with cosine period of 2 epochs compared
# # - 3hrs on 4GPU (or 12hrs on 1GPU) compared to 24hrs on 4GPU
# # - updated 4x batch size, 4x hidden dimension, afCRPS vs CRPS for ViT
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run diff_cns64_flow_matching_vit_55a4d1a_511e7b1 "FM (12hrs, old optim)" \
# 	--run diff_cns64_flow_matching_vit_2007857_2cfb01f "FM (3hrs)" \
# 	--run diff_cns64_flow_matching_vit_1ed9013_893624d "FM (24hrs)" \
# 	--run crps_cns64_vit_azula_large_8fe25aa_d105b90 "ViT (12hrs, 1GPU, n=256, CRPS, h=768, old optim)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (3hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_ab31602 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run diff_sw2d64_flow_matching_vit_4d5fcbd_067529b "FM (12hrs, old optim)" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM (3hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_1ed9013_5529d56 "FM (24hrs)" \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (12hrs, 1GPU, n=256, CRPS, h=768, old optim)" \
# 	--run crps_cns64_vit_azula_large_1ed9013_babdaa8 "ViT (3hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_sw2d64_vit_azula_large_1ed9013_6ab9fa2 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--dataset-order SW, CNS\
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/optimizer_24hrs_crps-changes_comparison

# # Revisit b16-mixed with lower batch size and updated optimizers:
# # TODO: add once eval completed
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_sw2d64_vit_azula_large_8fe25aa_74f91d8 "ViT (12hrs, 1GPU, n=256, CRPS, h=768, old optim)" \
# 	--run crps_sw2d64_vit_azula_large_1ed9013_6ab9fa2 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run crps_sw2d64_vit_azula_large_ed15816_085752e "ViT (6hrs, n=1024, afCRPS, h=632, b16-mixed, bs=32)" \
# 	--dataset-order SW\
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/b16-mixed_lower_bs_comparison

# # SW in ambient compared to latent
# autocast-plots --results-dir outputs/2026-04-01_collated \
# 	--run crps_sw2d64_vit_azula_large_1ed9013_6ab9fa2 "ViT (24hrs, n=1024, afCRPS, h=632)" \
# 	--run diff_sw2d64_flow_matching_vit_cb09424_7566c5e "FM (3hrs)" \
# 	--run diff_sw2d64_flow_matching_vit_53e8e0e_933bcfa "FM (3hrs ambient, new optim)" \
# 	--run diff_sw2d64_flow_matching_vit_53e8e0e_933bcfa_ema "FM (3hrs ambient, new optim, EMA)" \
# 	--dataset-order SW\
# 	--color-by-label \
# 	--error-ylim 1e-3 1 \
# 	--lead-time-error-metrics vrmse rmse \
# 	--lead-time-coverage-metrics coverage_0.9 coverage_0.5 coverage_0.1 \
# 	--combined-lead-time \
# 	--training-metrics val_loss train_loss \
# 	--training-yscale log \
# 	--panel-figure \
# 	--output-dir outputs/2026-04-01_collated/$PLOTS_PATH/sw_ambient_vs_latent
