#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
export ADDITIONAL_ARGS=$4
uv run python -m autocast.train.encoder_processor_decoder \
	--config-path=configs \
	--config-name=encoder_processor_decoder \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	data=$DATAPATH \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false \
	model.learning_rate=0.0002 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true \
	training.autoencoder_checkpoint=outputs/${LABEL}/${OUTPATH}/autoencoder.ckpt $ADDITIONAL_ARGS

