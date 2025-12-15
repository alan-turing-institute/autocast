#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 3:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 36
#SBATCH --job-name train_and_eval_autoencoder

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0

# Activate virtual environment
cd autocast # Change to the directory where your code is located
source venv/bin/activate

# Pip install to get current version of code 
uv sync --extra dev

# Run Autocast Code 

# First define a timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="encoder_processor_decoder_run"
WORKING_DIR="outputs/${JOB_NAME}/${TIMESTAMP}"

# Train
uv run train_encoder_processor_decoder \
    --config-path=configs/ \
	--work-dir=${WORKING_DIR}
	
# Evaluate
uv run evaluate_encoder_processor_decoder \
	--config-path=configs/ \
	--work-dir=${WORKING_DIR} \
	--checkpoint=${WORKING_DIR}/encoder_processor_decoder.ckpt \
	--batch-index=0 --batch-index=3 \
	--video-dir=${WORKING_DIR}/videos