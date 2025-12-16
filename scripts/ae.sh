#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
uv run python -m autocast.train.autoencoder \
	--config-path=configs \
	--config-name=ae_${LABEL} \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false \
	model.learning_rate=0.00005 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true
