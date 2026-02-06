#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name ae
#SBATCH --output=logs/ae%j.out
#SBATCH --error=logs/ae%j.err

# This forces script to fail as soon as it hits an error
set -e

# Pip install to get current version of code    
uv sync --extra dev

# Activate virtual environment - This assumes you have already created a virtual environment in the project directory
# If you haven't, replace with `uv venv`
source .venv/bin/activate

# First define a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Now define a job name. This will be used to create a unique working directory for outputs.
# Change this as needed
JOB_NAME="processor"

# This builds the working directory path. 
# It follows the structure outputs/JOB_NAME/TIMESTAMP
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"

# Write the slurm output and error files to the working directory
mkdir -p "${WORKING_DIR}"

exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# ---------------- Code to train and evaluate the model ----------------

export DATAPATH="advection_diffusion_multichannel"
export USE_NORMALIZATION="true"

# Train
srun uv run train_autoencoder \
    hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
	trainer.max_epochs=100 \
	logging.wandb.enabled=true \
	optimizer.learning_rate=0.00002

