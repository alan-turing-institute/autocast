#!/bin/bash

set -e

export DATASET="ad"
export OUTPATH="epd_00"
export DATAPATH="advection_diffusion_singlechannel"
uv run python -m autocast.train.autoencoder \
	--config-path=configs \
    --config-name=ae_${DATASET} \
	--work-dir=outputs/${DATASET}/${OUTPATH} \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false \
	model.learning_rate=0.00005 \
	trainer.max_epochs=20 \
	logging.wandb.enabled=true
    