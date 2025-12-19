#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3

uv run python -m autocast.train.processor \
	--config-path=configs \
	--config-name=processor \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	model.learning_rate=0.0005 \
	trainer.max_epochs=10 \
	logging.wandb.enabled=true
