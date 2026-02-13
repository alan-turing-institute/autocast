#!/bin/bash
# Sweep launcher examples using Hydra --multirun with submitit SLURM launcher.
#
# Prerequisites:
#   uv add hydra-submitit-launcher
#
# Usage:
#   Uncomment one of the examples below, adjust parameters, and run:
#     bash scripts/sweep.sh
#
# Each command submits one SLURM job per parameter combination automatically.
# No sbatch needed — submitit handles submission.

set -e

export AUTOCAST_DATASETS="${AUTOCAST_DATASETS:-$PWD/datasets}"

DATAPATH="advection_diffusion_multichannel_64_64" # Options: "advection_diffusion_multichannel_64_64", "advection_diffusion_multichannel"

# ---- Autoencoder sweep ----
# uv run train_autoencoder --multirun \
#     hydra/launcher=slurm \
#     datamodule=${DATAPATH} \
#     datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
#     datamodule.use_normalization=false \
#     encoder@model.encoder=dc_deep_256_v2 \
#     decoder@model.decoder=dc_deep_256_v2 \
#     trainer.max_epochs=100 \
#     logging.wandb.enabled=true

# ---- EPD diffusion sweep (hidden dim) ----
# uv run train_encoder_processor_decoder --multirun \
#     hydra/launcher=slurm \
#     datamodule=${DATAPATH} \
#     datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
#     datamodule.use_normalization=false \
#     processor@model.processor=flow_matching_vit \
#     encoder@model.encoder=dc_deep_256 \
#     decoder@model.decoder=dc_deep_256 \
#     model.train_in_latent_space=true \
#     +autoencoder_checkpoint=/path/to/autoencoder.ckpt \
#     model.processor.backbone.hid_channels=256,512,1024 \
#     optimizer.learning_rate=0.0002 \
#     datamodule.batch_size=128 \
#     trainer.max_epochs=200 \
#     logging.wandb.enabled=true

# ---- EPD CRPS sweep (ViT, hidden dim × noise type) ----
# NOTE: CLN NOISE IS HANDLED DIFFERENTLY
# uv run train_encoder_processor_decoder --multirun \
#     hydra/launcher=slurm \
#     datamodule=${DATAPATH} \
#     datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
#     datamodule.use_normalization=false \
#     processor@model.processor=vit_large \
#     encoder@model.encoder=permute_concat \
#     model.encoder.with_constants=true \
#     decoder@model.decoder=channels_last \
#     model.processor.hidden_dim=256,512 \
#     model.processor.spatial_resolution="[64,64]" \
#     model.processor.patch_size=null \
#     input_noise_injector@model.input_noise_injector=concat,additive \
#     model.train_in_latent_space=false \
#     +model.n_members=10 \
#     "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss" \
#     "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS" \
#     optimizer.learning_rate=0.0002 \
#     datamodule.batch_size=64 \
#     trainer.max_epochs=100 \
#     logging.wandb.enabled=true

# ---- EPD CRPS sweep (FNO, hidden dim × noise type) ----
# uv run train_encoder_processor_decoder --multirun \
#     hydra/launcher=slurm \
#     datamodule=${DATAPATH} \
#     datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
#     datamodule.use_normalization=false \
#     processor@model.processor=fno \
#     encoder@model.encoder=permute_concat \
#     model.encoder.with_constants=true \
#     decoder@model.decoder=channels_last \
#     model.processor.hidden_channels=256,512 \
#     input_noise_injector@model.input_noise_injector=concat,additive \
#     model.train_in_latent_space=false \
#     +model.n_members=10 \
#     "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss" \
#     "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS" \
#     optimizer.learning_rate=0.0002 \
#     datamodule.batch_size=16 \
#     trainer.max_epochs=100 \
#     logging.wandb.enabled=true
