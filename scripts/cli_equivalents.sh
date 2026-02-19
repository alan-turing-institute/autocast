#!/bin/bash

set -euo pipefail

# CLI equivalents of removed slurm_scripts examples.
# Usage:
#   bash scripts/cli_equivalents.sh
#
# Notes:
# - These examples submit non-blocking train/eval chains on SLURM via --detach.
# - Date/run_name follow legacy pattern: outputs/<date>/<run_name>

export AUTOCAST_DATASETS="${AUTOCAST_DATASETS:-$PWD/datasets}"
TODAY="$(date +%F)"
GIT_HASH="$(git rev-parse --short=7 HEAD | tr -d '\n')"
UUID="$(uuidgen | tr -d '\n' | tail -c 7)"

# -----------------------------------------------------------------------------
# 1) train_and_eval_autoencoder_64_64_dc_deep_256_v2_no_norm.sh
# -----------------------------------------------------------------------------
AE_DATAPATH="advection_diffusion_multichannel_64_64"
AE_MODEL="flow_matching_vit"
AE_RUN_NAME="ae_${AE_DATAPATH}_${AE_MODEL}_${GIT_HASH}_${UUID}"

echo "# Autoencoder equivalent"
echo "uv run autocast ae --mode slurm --dataset ${AE_DATAPATH} --date ${TODAY} --run-name ${AE_RUN_NAME} datamodule.use_normalization=false logging.wandb.enabled=true logging.wandb.name=${AE_RUN_NAME} trainer.max_epochs=200 optimizer.learning_rate=0.00002 encoder@model.encoder=dc_deep_256_v2 decoder@model.decoder=dc_deep_256_v2"

# -----------------------------------------------------------------------------
# 2) train_and_eval_epd_crps_fno_additive.sh
# -----------------------------------------------------------------------------
CRPS_DATAPATH="advection_diffusion_multichannel_64_64"
CRPS_MODEL="fno"
CRPS_NOISE="additive"
CRPS_HIDDEN=256
CRPS_RUN_NAME="crps_${CRPS_DATAPATH}_${CRPS_MODEL}_${CRPS_NOISE}_${CRPS_HIDDEN}_${GIT_HASH}_${UUID}"

CRPS_COMMON=(
  datamodule.use_normalization=false
  logging.wandb.enabled=true
  "logging.wandb.name=${CRPS_RUN_NAME}"
  optimizer.learning_rate=0.0002
  encoder@model.encoder=permute_concat
  model.encoder.with_constants=true
  decoder@model.decoder=channels_last
  processor@model.processor=fno
  model.processor.hidden_channels=256
  input_noise_injector@model.input_noise_injector=additive
  datamodule.batch_size=16
  trainer.max_epochs=100
  model.train_in_latent_space=false
  +model.n_members=10
  model.loss_func._target_=autocast.losses.ensemble.CRPSLoss
  +model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS
)

echo
echo "# EPD CRPS FNO additive equivalent"
echo "uv run autocast train-eval --mode slurm --detach --dataset ${CRPS_DATAPATH} --date ${TODAY} --run-name ${CRPS_RUN_NAME} ${CRPS_COMMON[*]} ::eval:: ${CRPS_COMMON[*]} eval.batch_indices=[0,1,2,3]"

# -----------------------------------------------------------------------------
# 3) train_and_eval_epd_diffusion_flow_matching_200.sh
# -----------------------------------------------------------------------------
DIFF_DATAPATH="advection_diffusion_multichannel_64_64"
DIFF_MODEL="flow_matching_vit"
DIFF_HIDDEN=512
DIFF_RUN_NAME="diff_${DIFF_DATAPATH}_${DIFF_MODEL}_${DIFF_HIDDEN}_${GIT_HASH}_${UUID}"

# In old script this was symlinked into workdir. Here we pass path directly.
DIFF_AE_CHECKPOINT="/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_64_64_no_norm/autoencoder.ckpt"

DIFF_COMMON=(
  datamodule.use_normalization=false
  logging.wandb.enabled=true
  "logging.wandb.name=${DIFF_RUN_NAME}"
  processor@model.processor=flow_matching_vit
  datamodule.batch_size=128
  optimizer.learning_rate=0.0002
  encoder@model.encoder=dc_deep_256
  decoder@model.decoder=dc_deep_256
  model.train_in_latent_space=true
  model.processor.backbone.hid_channels=512
  trainer.max_epochs=200
  +autoencoder_checkpoint="${DIFF_AE_CHECKPOINT}"
)

echo
echo "# EPD diffusion flow-matching equivalent"
echo "uv run autocast train-eval --mode slurm --detach --dataset ${DIFF_DATAPATH} --date ${TODAY} --run-name ${DIFF_RUN_NAME} ${DIFF_COMMON[*]} ::eval:: ${DIFF_COMMON[*]} +model.n_members=10 eval.batch_indices=[0,1,2,3]"
