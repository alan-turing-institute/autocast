#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name ae
#SBATCH --output=logs/ae_%j.out
#SBATCH --error=logs/ae_%j.err

set -e

# Enable write permissions for group
umask 0002

# Might be used within python scripts
export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64" # Options: "advection_diffusion_multichannel_64_64", "advection_diffusion_multichannel"
USE_NORMALIZATION="false" # Options: "true" or "false"
MODEL="dc_deep_256_v2" # Options (any compatible config in configs/processors/), currently: "flow_matching_vit", "diffusion_vit"
EPOCHS=100

# One model params block for now since shared config pattern
MODEL_PARAMS=(
    "encoder@model.encoder=${MODEL}"
    "decoder@model.decoder=${MODEL}"
)

# Derive code and unique run identifiers
GIT_HASH=$(git rev-parse --short=7 HEAD | tr -d '\n')
UUID=$(uuidgen | tr -d '\n' | tail -c 7)

# Load dataset aliases
source "$PWD/slurm_templates/isambard/dataset_aliases.sh"

# Run name and working directory
RUN_NAME="ae_${DATA_SHORT}_${MODEL}_${GIT_HASH}_${UUID}"
WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"


# Make directories and redirect output and error logs to the working directory
mkdir -p $WORKING_DIR
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Training
# Train
srun uv run train_autoencoder \
     hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
	logging.wandb.enabled=true \
	logging.wandb.name="${RUN_NAME}" \
	"${MODEL_PARAMS[@]}"	
