#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
export ADDITIONAL_ARGS=$4
uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--config-name=encoder_processor_decoder \
	--work-dir=outputs/${LABEL}/${OUTPATH}/eval \
	--checkpoint=outputs/${LABEL}/${OUTPATH}/encoder_processor_decoder.ckpt \
	--batch-index=0 \
	--batch-index=1 \
	--batch-index=2 \
	--batch-index=3 \
	--video-dir=outputs/${LABEL}/${OUTPATH}/eval/videos \
	data=$DATAPATH \
	data.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	data.use_simulator=false $ADDITIONAL_ARGS
