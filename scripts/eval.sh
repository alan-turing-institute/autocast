#!/bin/bash

set -e

if [ "$#" -lt 3 ]; then
	echo "Usage: $0 <label> <run_id> <dataset> [overrides...]"
	exit 1
fi

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

export AUTOCAST_DATASETS="${AUTOCAST_DATASETS:-$PWD/datasets}"
LAUNCH_MODE="${LAUNCH_MODE:-local}" # local|slurm

BASE_DIR="${PWD}/outputs/${LABEL}/${OUTPATH}"
RUN_DIR="${BASE_DIR}/eval"
CKPT_PATH="${BASE_DIR}/encoder_processor_decoder.ckpt"
if [ "${LAUNCH_MODE}" = "slurm" ] && [ ! -f "${CKPT_PATH}" ]; then
	SLURM_CKPT="${BASE_DIR}/run/encoder_processor_decoder.ckpt"
	if [ -f "${SLURM_CKPT}" ]; then
		CKPT_PATH="${SLURM_CKPT}"
	fi
fi
VIDEO_DIR="${RUN_DIR}/videos"

OVERRIDES=(
	"eval=encoder_processor_decoder"
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
	"eval.checkpoint=${CKPT_PATH}"
	"eval.batch_indices=[0,1,2,3]"
	"eval.video_dir=${VIDEO_DIR}"
)

if [ "${LAUNCH_MODE}" = "slurm" ]; then
	OVERRIDES=(
		"hydra.mode=MULTIRUN"
		"hydra/launcher=slurm"
		"hydra.sweep.dir=${RUN_DIR}"
		"hydra.sweep.subdir=."
		"${OVERRIDES[@]}"
	)
else
	OVERRIDES=(
		"hydra.run.dir=${RUN_DIR}"
		"${OVERRIDES[@]}"
	)
fi

uv run evaluate_encoder_processor_decoder \
	"${OVERRIDES[@]}" \
	"$@"
