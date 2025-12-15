#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --job-name train_and_eval_autoencoder
#SBATCH --output=logs/train_and_eval_autoencoder_%j.out
#SBATCH --error=logs/train_and_eval_autoencoder_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0


# Pip install to get current version of code    
uv sync --extra dev

# Activate virtual environment - This assumes you have already created a virtual environment in the project directory
# If you haven't, replace with `uv venv`
source .venv/bin/activate
# First define a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Now define a job name. This will be used to create a unique working directory for outputs.
# Change this as needed
JOB_NAME="autoencoder_run"

# This builds the working directory path. 
# It follows the structure outputs/JOB_NAME/TIMESTAMP
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"


# Write the slurm output and error files to the working directory
mkdir -p "${WORKING_DIR}"

exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# ---------------- Code to train and evaluate the model ----------------

# Train
uv run train_autoencoder \
    --config-path=configs/ \
	--work-dir=${WORKING_DIR}
	

