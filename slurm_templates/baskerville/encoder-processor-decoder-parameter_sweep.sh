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

# ---------------- Define Parameter Grid ----------------

MAX_EPOCHS=(3 5 10)
BATCH_SIZES=(4 8 16)

N_MAX_EPOCHS=${#MAX_EPOCHS[@]}
N_BATCH_SIZES=${#BATCH_SIZES[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate indices for max epochs and batch size based on task ID
EPOCH_INDEX=$((SLURM_ARRAY_TASK_ID % N_MAX_EPOCHS))
BATCH_INDEX=$((SLURM_ARRAY_TASK_ID / N_MAX_EPOCHS))

# Get the parameters for this task
MAX_EPOCH=${MAX_EPOCHS[$EPOCH_INDEX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_INDEX]}
echo "Task ID: ${SLURM_ARRAY_TASK_ID}, Max Epochs: ${MAX_EPOCH}, Batch Size: ${BATCH_SIZE}"

# ---------------- Write Parameter Lookup ----------------
LOOKUP_DIR="outputs/${JOB_NAME}/job-${JOB_ID}"
LOOKUP_FILE="${LOOKUP_DIR}/parameter_lookup.csv"

# Create header for lookup file (only first task creates header)
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    echo "TaskID,MaxEpochs,BatchSize" > "${LOOKUP_FILE}"
fi

# Wait a moment to ensure header is written
sleep 1

# Append this task's parameters to the lookup file
echo "${SLURM_ARRAY_TASK_ID},${MAX_EPOCH},${BATCH_SIZE}" >> "${LOOKUP_FILE}"

# ---------------- Train and Evaluate Model ----------------
# Train
uv run train_encoder_processor_decoder \
	hydra.run.dir=${WORKING_DIR} \
	trainer.max_epochs=${MAX_EPOCH} \
	datamodule.batch_size=${BATCH_SIZE}

	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	hydra.run.dir=${WORKING_DIR} \
	eval=encoder_processor_decoder \
	eval.checkpoint=${WORKING_DIR}/autocast/*/checkpoints/last.ckpt \
	eval.batch_indices=[0,3] \
	eval.video_dir=${WORKING_DIR}/videos

