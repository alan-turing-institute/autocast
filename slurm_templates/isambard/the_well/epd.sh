#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --job-name processor
#SBATCH --output=logs/processor_%j.out
#SBATCH --error=logs/processor_%j.err

# This forces script to fail as soon as it hits an error
set -e

# Load necessary modules. Adjust as needed for your environment.
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
JOB_NAME="processor"

# This builds the working directory path. 
# It follows the structure outputs/JOB_NAME/TIMESTAMP
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"

# Write the slurm output and error files to the working directory
mkdir -p "${WORKING_DIR}"

exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"

# Mitigate CUDA fragmentation when memory is tight
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Backwards compatibility for older PyTorch builds
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------- Code to train and evaluate the model ----------------

# Train
srun uv run train_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
    model=encoder_processor_decoder \
    encoder@model.encoder=dc_f32c64_small \
    decoder@model.decoder=dc_f32c64_small \
    processor@model.processor=flow_matching_large \
    datamodule=the_well \
    datamodule.well_dataset_name=rayleigh_benard \
    datamodule.num_workers=0 \
    datamodule.batch_size=8 \
    datamodule.cache_small=false \
    datamodule.max_cache_size=0.0 \
    optimizer=adamw \
    logging.wandb.enabled=true \
    trainer.max_epochs=5 \
    trainer.gradient_clip_val=1.0 \
    trainer.callbacks.0.every_n_train_steps=5000 \
    "autoencoder_checkpoint='outputs/autoencoder_run/20251217_121300/autocast/0nttzj9a/checkpoints/step-step=7900.ckpt'"
	# trainer.enable_checkpointing=false \

# Evaluate
srun uv run evaluate_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
    eval=encoder_processor_decoder \
	eval.checkpoint=${WORKING_DIR}/autocast/*/checkpoints/last.ckpt \
	eval.batch_indices=[0,1] \
	eval.video_dir=${WORKING_DIR}/videos