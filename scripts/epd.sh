#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

WORKDIR=outputs/${LABEL}/${OUTPATH}

# Build overrides and include the autoencoder checkpoint only if present
OVERRIDES=(
	--config-path=configs
	--config-name=encoder_processor_decoder
	--work-dir=${WORKDIR}
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
)

CKPT="${WORKDIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
	OVERRIDES+=( "+autoencoder_checkpoint=${CKPT}" )
fi

# Run script
# Optional overrides you can add via CLI:
#   logging.wandb.enabled=true
uv run python -m autocast.scripts.train.encoder_processor_decoder "${OVERRIDES[@]}" "$@"

# Example Usage for Ensemble via CLI Args:
# ./scripts/epd.sh my_label my_run reaction_diffusion \
#     encoder@model.encoder=permute_concat \
#     decoder@model.decoder=channels_last \
#     processor@model.processor=vit \
#     model.processor.n_noise_channels=1000 \
#     +model.n_members=10 \
#     model.loss_func._target_=autocast.losses.ensemble.CRPSLoss \
#     +model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS \
#     logging.wandb.enabled=true \
#     model.learning_rate=0.0002 \
#     trainer.max_epochs=5