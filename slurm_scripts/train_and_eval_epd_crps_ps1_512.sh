#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name epd_and_eval
#SBATCH --output=logs/epd_and_eval_%j.out
#SBATCH --error=logs/epd_and_eval_%j.err

set -e

# Enable write permissions for group
umask 0002

# Might be used within python scripts
export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64" # Options: "advection_diffusion_multichannel_64_64", "advection_diffusion_multichannel"
USE_NORMALIZATION="false" # Options: "true" or "false"
MODEL="vit_large" # Options (any compatible config in configs/processors/), currently: "vit", "vit_large", "fno"
HIDDEN_DIM=512 # Any positive integer, e.g. 256, 512, 1024, etc.
MODEL_NOISE="cln" # Options: "cln", "concat", "additive"
EPOCHS=12
EVAL_BATCH_SIZE=8
LEARNING_RATE=0.0002
EVAL_ONLY="false"
WORKING_DIR=""

# These assume a single noise per spatial point (not per time step).
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    NOISE_CHANNELS=4096
elif [ ${DATAPATH} == "advection_diffusion_multichannel" ]; then
    NOISE_CHANNELS=1024
fi

# Hidden dimension parameters
if [ ${MODEL} == "vit_large" ]; then
    HIDDEN_PARAMS="model.processor.hidden_dim=${HIDDEN_DIM}"
else
    HIDDEN_PARAMS="model.processor.hidden_channels=${HIDDEN_DIM}"
fi

# Input noise injection
if [ ${MODEL_NOISE} == "cln" ]; then
    MODEL_NOISE_PARAMS="model.processor.n_noise_channels=${NOISE_CHANNELS}"
elif [ ${MODEL_NOISE} == "concat" ]; then
    MODEL_NOISE_PARAMS="input_noise_injector@model.input_noise_injector=concat"
elif [ ${MODEL_NOISE} == "additive" ]; then
    MODEL_NOISE_PARAMS="input_noise_injector@model.input_noise_injector=additive"
fi

# Spatial resolution parameters
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[64,64]"
else
    SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[32,32]"
fi

if [ ${MODEL} == "vit_large" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${MODEL_NOISE_PARAMS}"
        "${SPATIAL_RESOLUTION_PARAMS}"
        "${HIDDEN_PARAMS}"
        "model.processor.patch_size=1"
		"datamodule.batch_size=8"
    )
elif [ ${MODEL} == "fno" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${HIDDEN_PARAMS}"
        "${MODEL_NOISE_PARAMS}"
		"datamodule.batch_size=16"
    )
fi

# Combine all model parameters
MODEL_PARAMS=(
     "optimizer.learning_rate=${LEARNING_RATE}"
     "encoder@model.encoder=permute_concat"
     "model.encoder.with_constants=true"
     "decoder@model.decoder=channels_last"
)
MODEL_PARAMS+=("${MODEL_SPECIFIC_PARAMS[@]}")
MODEL_PARAMS+=(
    "trainer.max_epochs=${EPOCHS}"
	 "model.train_in_latent_space=false"
     "+model.n_members=10"
     "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss"
     "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS"
)

# Derive code and unique run identifiers
GIT_HASH=$(git rev-parse --short=7 HEAD | tr -d '\n')
UUID=$(uuidgen | tr -d '\n' | tail -c 7)

# Load dataset aliases
source "$PWD/slurm_templates/isambard/dataset_aliases.sh"

# Run name and working directory
RUN_NAME="crps_${DATA_SHORT}_${MODEL}_${MODEL_NOISE}_${HIDDEN_DIM}_${GIT_HASH}_${UUID}"

if [ ${EVAL_ONLY} = "false" ]; then
	WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"
else
	if [ "${WORKING_DIR}" = "" ]; then
		echo "Error: WORKING_DIR must be set when EVAL_ONLY is true."
		exit 1
	fi
fi

# Check if there's a pretrained autoencoder checkpoint in the working directory
CKPT="${WORKING_DIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
        MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

# Make directories and redirect output and error logs to the working directory
if [ ${EVAL_ONLY} = "false" ]; then
	mkdir -p $WORKING_DIR
fi
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Training
if [ ${EVAL_ONLY} = "false" ]; then
	srun uv run train_encoder_processor_decoder \
		hydra.run.dir=${WORKING_DIR} \
		datamodule="${DATAPATH}" \
		datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
		datamodule.use_normalization="${USE_NORMALIZATION}" \
		logging.wandb.enabled=true \
		logging.wandb.name="${RUN_NAME}" \
		"${MODEL_PARAMS[@]}"
fi

# Eval
CKPT_PATH="${WORKING_DIR}/encoder_processor_decoder.ckpt"
EVAL_DIR="${WORKING_DIR}/eval"

srun uv run evaluate_encoder_processor_decoder \
    hydra.run.dir="${EVAL_DIR}" \
    eval=encoder_processor_decoder \
    datamodule="${DATAPATH}" \
    datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
    eval.checkpoint=${CKPT_PATH} \
    eval.batch_indices=[0,1,2,3,4,5,6,7] \
    eval.video_dir="${EVAL_DIR}/videos" \
    "${MODEL_PARAMS[@]}" \
    datamodule.batch_size=${EVAL_BATCH_SIZE}
