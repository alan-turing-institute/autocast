#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --job-name train_and_eval_encoder_processor_decoder
#SBATCH --output=outputs/logs/train_and_eval_encoder_processor_decoder_%j.out
#SBATCH --error=outputs/logs/train_and_eval_encoder_processor_decoder_%j.err

set -e

# First define a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Now define a job name. This will be used to create a unique working directory for outputs.
# Change this as needed
JOB_NAME="encoder_processor_decoder_run"

# This builds the working directory path. 
# It follows the structure outputs/JOB_NAME/TIMESTAMP
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"


# Write the slurm output and error files to the working directory
mkdir -p "${WORKING_DIR}"

exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# ---------------- Load modules and activate environment ----------------
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

# Mitigate CUDA fragmentation when memory is tight
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Backwards compatibility for older PyTorch builds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------- Code to train and evaluate the model ----------------

# Train
 uv run train_encoder_processor_decoder \
      --config-path=configs/ \
          --config-name=encoder_processor_decoder \
          model=encoder_processor_decoder \
          encoder@model.encoder=permute_concat \
          decoder@model.decoder=channels_last \
          processor@model.processor=fno \
          logging.wandb.enabled=true \
          trainer.max_epochs=5 \
          data=the_well \
          data.well_dataset_name=turbulent_radiative_layer_2D \
          data.num_workers=0 \
          data.batch_size=4 \
          optimizer=adamw \
          optimizer.learning_rate=0.0001 \
          optimizer.weight_decay=1e-4

uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--work-dir=${WORKING_DIR} \
	--checkpoint=${WORKING_DIR}/encoder_processor_decoder.ckpt \
	--batch-index=0 \
    --batch-index=1 \
	--video-dir=${WORKING_DIR}/videos