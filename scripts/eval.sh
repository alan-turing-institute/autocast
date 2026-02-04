#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

# Run script
RUN_DIR="${PWD}/outputs/${LABEL}/${OUTPATH}/eval"
CKPT_PATH="${PWD}/outputs/${LABEL}/${OUTPATH}/encoder_processor_decoder.ckpt"
VIDEO_DIR="${RUN_DIR}/videos"

uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=${RUN_DIR} \
	eval=encoder_processor_decoder \
	datamodule=${DATAPATH} \
	datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH} \
	eval.checkpoint=${CKPT_PATH} \
	eval.video_dir=${VIDEO_DIR} \
	"$@"
