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

WORKDIR="${PWD}/outputs/${LABEL}/${OUTPATH}"

OVERRIDES=(
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
)

if [ "${LAUNCH_MODE}" = "slurm" ]; then
	OVERRIDES=(
		"hydra.mode=MULTIRUN"
		"hydra/launcher=slurm"
		"hydra.sweep.dir=${WORKDIR}"
		"hydra.sweep.subdir=."
		"${OVERRIDES[@]}"
	)
else
	OVERRIDES=(
		"hydra.run.dir=${WORKDIR}"
		"${OVERRIDES[@]}"
	)
fi

# Run script
# Optional overrides you can add via CLI:
#   logging.wandb.enabled=true
uv run python -m autocast.scripts.train.processor "${OVERRIDES[@]}" "$@"
    
