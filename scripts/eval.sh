#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

# Run script
uv run python -m autocast.scripts.eval.encoder_processor_decoder \
	--config-path=configs/ \
	--config-name=encoder_processor_decoder \
	--work-dir=outputs/${LABEL}/${OUTPATH}/eval \
	--checkpoint=outputs/${LABEL}/${OUTPATH}/encoder_processor_decoder.ckpt \
	--batch-index=0 \
	--batch-index=1 \
	--batch-index=2 \
	--batch-index=3 \
	--video-dir=outputs/${LABEL}/${OUTPATH}/eval/videos \
	datamodule=${DATAPATH} \
	datamodule.data_path=$AUTOCAST_DATASETS/${DATAPATH} \
	"$@"
