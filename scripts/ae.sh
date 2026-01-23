#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
export ADDITIONAL_ARGS=$4
uv run python -m autocast.train.autoencoder \
	--config-path=configs \
	--config-name=autoencoder \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	data=$DATAPATH \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false \
	model.learning_rate=0.00002 \
	trainer.max_epochs=20 \
	logging.wandb.enabled=true $ADDITIONAL_ARGS
