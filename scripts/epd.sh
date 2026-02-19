#!/bin/bash

set -e

if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <label> <run_id> <dataset> [overrides...]"
	exit 1
fi

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

export AUTOCAST_DATASETS="${AUTOCAST_DATASETS:-$PWD/datasets}"
LAUNCH_MODE="${LAUNCH_MODE:-local}" # local|slurm

WORKDIR="${PWD}/outputs/${LABEL}/${OUTPATH}"

# Build overrides and include the autoencoder checkpoint only if present
OVERRIDES=(
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
)

CKPT="${WORKDIR}/autoencoder.ckpt"
if [ "${LAUNCH_MODE}" = "slurm" ] && [ ! -f "${CKPT}" ]; then
	SLURM_CKPT="${WORKDIR}/run/autoencoder.ckpt"
	if [ -f "${SLURM_CKPT}" ]; then
		CKPT="${SLURM_CKPT}"
	fi
fi
if [ -f "${CKPT}" ]; then
	OVERRIDES+=( "+autoencoder_checkpoint=${CKPT}" )
fi

if [ "${LAUNCH_MODE}" = "slurm" ]; then
	OVERRIDES=(
		"hydra.mode=MULTIRUN"
		"hydra/launcher=slurm"
		"hydra.sweep.dir=${WORKDIR}"
		"hydra.sweep.subdir=."
		"${OVERRIDES[@]}"
	)
else
	OVERRIDES=(
		"hydra.run.dir=${WORKDIR}"
		"${OVERRIDES[@]}"
	)
fi

# Run script
# Optional overrides you can add via CLI:
#   logging.wandb.enabled=true
uv run train_encoder_processor_decoder "${OVERRIDES[@]}" "$@"

# Example Usage for Ensemble via CLI Args:
# ./scripts/epd.sh my_label my_run reaction_diffusion \
#     encoder@model.encoder=permute_concat \
#     model.encoder.with_constants=true \
#     decoder@model.decoder=channels_last \
#     processor@model.processor=vit \
#     model.processor.n_noise_channels=1000 \
#     +model.n_members=10 \
#     model.loss_func._target_=autocast.losses.ensemble.CRPSLoss \
#     +model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS \
#     logging.wandb.enabled=false \
#     optimizer.learning_rate=0.0002 \
#     trainer.max_epochs=5
