#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
export ADDITIONAL_ARGS=$4

# Run script
uv run python -m autocast.scripts.train.autoencoder \
	--config-path=configs \
	--config-name=autoencoder \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	datamodule=${DATAPATH} \
	datamodule.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	model.learning_rate=0.00002 \
	trainer.max_epochs=20 \
	logging.wandb.enabled=true $ADDITIONAL_ARGS
