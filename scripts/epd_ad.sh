#!/bin/bash


set -e

export DATASET="ad"
export OUTPATH="epd_00"
export DATAPATH="advection_diffusion_singlechannel"

uv run python -m autocast.train.encoder_processor_decoder \
	--config-path=configs \
    --config-name=epd_${DATASET} \
	--work-dir=outputs/${DATASET}/${OUTPATH} \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false \
	model.learning_rate=0.0005 \
	trainer.max_epochs=30 \
	logging.wandb.enabled=true \
	training.autoencoder_checkpoint=outputs/${DATASET}/${OUTPATH}/autoencoder.ckpt