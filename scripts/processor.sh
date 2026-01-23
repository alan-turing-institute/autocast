#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
export ADDITIONAL_ARGS=$4
uv run python -m autocast.train.processor \
	--config-path=configs \
	--config-name=processor \
	--work-dir=outputs/${LABEL}/${OUTPATH} \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	model.learning_rate=0.0001 \
	trainer.max_epochs=10 \
    processor@model.processor=diffusion_vit \
	backbone@model.processor.backbone=vit_small \
	data.batch_size=16 \
	logging.wandb.enabled=true \
	+trainer.limit_train_batches=0.2 \
	+trainer.limit_val_batches=0.1 \
    +trainer.limit_test_batches=0.1 $ADDITIONAL_ARGS
    
