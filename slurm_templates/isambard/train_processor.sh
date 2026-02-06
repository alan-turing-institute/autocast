#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name well-benchmark
#SBATCH --output=logs/well-benchmark%j.out
#SBATCH --error=logs/well-benchmark%j.err


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

# Train
srun uv run train_processor \
    hydra.run.dir=${WORKING_DIR} \
    datamodule.data_path=datasets/rayleigh_benard/1e3z5x2c_rayleigh_benard_dcae_f32c64_large/cache/rayleigh_benard \
    datamodule.batch_size=64 \
    datamodule.n_steps_input=1 \
    datamodule.n_steps_output=4 \
    processor@model.processor=diffusion_vit \
    datamodule.stride=1 \
    trainer.max_epochs=20 \
    logging.wandb.enabled=true
	

