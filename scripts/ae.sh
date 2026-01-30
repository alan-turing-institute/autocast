#!/bin/bash

set -e

export LABEL=$1
export OUTPATH=$2
export DATAPATH=$3
shift 3

WORKDIR=outputs/${LABEL}/${OUTPATH}

OVERRIDES=(
	--config-path=configs
	--config-name=autoencoder
	--work-dir=${WORKDIR}
	"datamodule=${DATAPATH}"
	"datamodule.data_path=${AUTOCAST_DATASETS}/${DATAPATH}"
)

# Run script
# Optional overrides you can add via CLI:
#   logging.wandb.enabled=true
uv run python -m autocast.scripts.train.autoencoder "${OVERRIDES[@]}" "$@"
