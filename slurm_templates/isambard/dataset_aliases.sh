#!/bin/bash

# Create a lookup dictionary for shorter dataset names
declare -A DATASET_ALIASES
DATASET_ALIASES=(
    ["advection_diffusion_multichannel_64_64"]="adm64"
    ["advection_diffusion_multichannel"]="adm32"
)

# Get alias or default to full name if not found
DATA_SHORT="${DATASET_ALIASES[$DATAPATH]}"
if [ -z "$DATA_SHORT" ]; then
    DATA_SHORT="${DATAPATH}"
fi
