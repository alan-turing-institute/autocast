#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00        
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --job-name epd_and_eval
#SBATCH --output=logs/epd_and_eval%j.out
#SBATCH --error=logs/epd_and_eval%j.err

set -e

export AUTOCAST_DATASETS="$PWD/datasets"

# Set configuration parameters
DATAPATH="advection_diffusion_multichannel_64_64"
USE_NORMALIZATION="false"
MODEL="vit_large"
HIDDEN_DIM=512
MODEL_NOISE="cln" # Options: "cln", "concat"

# Derive remaining parameters based on the dataset and model choices
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    NOISE_CHANNELS=4096
else
    NOISE_CHANNELS=1024
fi
RUN_NAME="${DATAPATH}_${MODEL}_${MODEL_NOISE}_${HIDDEN_DIM}_crps_no_norm"
WORKING_DIR="$PWD/outputs/$(date +%F)/${RUN_NAME}/"

mkdir -p $WORKING_DIR
exec > "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" \
     2> "${WORKING_DIR}/slurm_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"


# Flow matching
# MODEL_PARAMS=(
#     "optimizer.learning_rate=0.0002"
#     "encoder@model.encoder=dc_deep_256"
#     "decoder@model.decoder=dc_deep_256"
# )

# Hidden dimension parameters
if [ ${MODEL} == "vit_large" ]; then
    HIDDEN_PARAMS="model.processor.hidden_dim=${HIDDEN_DIM}"
else
    HIDDEN_PARAMS="model.processor.hidden_channels=${HIDDEN_DIM}"
fi

# Input noise injection
if [ ${MODEL_NOISE} == "cln" ]; then
    MODEL_NOISE_PARAMS="model.processor.n_noise_channels=${NOISE_CHANNELS}"
else
    MODEL_NOISE_PARAMS="input_noise_injector@model.input_noise_injector=concat"
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
        "model.processor.patch_size=null"
    )
elif [ ${MODEL} == "fno" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${HIDDEN_PARAMS}"
        "${MODEL_NOISE_PARAMS}"
    )
fi

# Combine all model parameters
MODEL_PARAMS=(
     "optimizer.learning_rate=0.0002"
     "encoder@model.encoder=permute_concat"
     "model.encoder.with_constants=true"
     "decoder@model.decoder=channels_last"
)
MODEL_PARAMS+=("${MODEL_SPECIFIC_PARAMS[@]}")
MODEL_PARAMS+=(
	 "model.train_in_latent_space=false"
     "+model.n_members=10"
     "model.loss_func._target_=autocast.losses.ensemble.CRPSLoss"
     "+model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS"
)

CKPT="${WORKING_DIR}/autoencoder.ckpt"
if [ -f "${CKPT}" ]; then
        MODEL_PARAMS+=( "+autoencoder_checkpoint=${CKPT}" )
fi

srun uv run train_encoder_processor_decoder \
    hydra.run.dir=${WORKING_DIR} \
	datamodule="${DATAPATH}" \
	datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
	datamodule.use_normalization="${USE_NORMALIZATION}" \
    datamodule.batch_size=128 \
	logging.wandb.enabled=true \
	logging.wandb.name="${RUN_NAME}" \
	 "${MODEL_PARAMS[@]}"
	
	
	
CKPT_PATH="${WORKING_DIR}/encoder_processor_decoder.ckpt"
EVAL_DIR="${WORKING_DIR}/eval"

srun uv run evaluate_encoder_processor_decoder \
    hydra.run.dir="${EVAL_DIR}" \
    eval=encoder_processor_decoder \
    datamodule="${DATAPATH}" \
    datamodule.data_path="${AUTOCAST_DATASETS}/${DATAPATH}" \
    datamodule.batch_size=128 \
    eval.checkpoint=${CKPT_PATH} \
    eval.batch_indices=[0,1,2,3] \
    eval.video_dir="${EVAL_DIR}/videos" \
    "${MODEL_PARAMS[@]}"
