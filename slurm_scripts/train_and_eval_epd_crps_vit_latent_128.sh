#!/bin/bash
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00        
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
MODEL="vit_latent" # Options (any compatible config in configs/processors/), currently: "vit", "vit_large", "fno"
HIDDEN_DIM=128 # Any positive integer, e.g. 256, 512, 1024, etc.
PATCH_SIZE=1 # Any positive integer that divides the spatial dimensions, e.g. 4, 8, 16, etc.
MODEL_NOISE="cln" # Options: "cln", "concat", "additive"
EPOCHS=35
EVAL_BATCH_SIZE=16
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
if [ ${MODEL} == "vit_latent" ]; then
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

# # Spatial resolution parameters
# if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
#     SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[64,64]"
# else
#     SPATIAL_RESOLUTION_PARAMS="model.processor.spatial_resolution=[32,32]"
# fi

# Autoencoder checkpoint paths based on dataset
if [ ${DATAPATH} == "advection_diffusion_multichannel_64_64" ]; then
    AE_CHECKPOINT="/home/u5gf/ltcx7228.u5gf/autocast/outputs/autoencoders/adm_64_1000.ckpt"
elif [ ${DATAPATH} == "advection_diffusion_multichannel" ]; then
    AE_CHECKPOINT="/projects/u5gf/ai4physics/outputs/2026-02-06/advection_diffusion_multichannel_no_norm/autoencoder.ckpt"
fi

if [ ${MODEL} == "vit_latent" ]; then
    MODEL_SPECIFIC_PARAMS=(
        "processor@model.processor=${MODEL}"
        "${MODEL_NOISE_PARAMS}"
        "${HIDDEN_PARAMS}"
		"datamodule.batch_size=16"
        "model.processor.patch_size=${PATCH_SIZE}"
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
    "encoder@model.encoder=dc_deep_256_v2"
    "decoder@model.decoder=dc_deep_256_v2"
    "model.train_in_latent_space=true"
)
MODEL_PARAMS+=("${MODEL_SPECIFIC_PARAMS[@]}")
MODEL_PARAMS+=(
    "trainer.max_epochs=${EPOCHS}"
	"model.train_in_latent_space=false"
    "model.freeze_encoder_decoder=true"
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
if [ ${EVAL_ONLY} = "false" ]; then
	mkdir -p "${WORKING_DIR}"
	cd "${WORKING_DIR}"
	ln -s "${AE_CHECKPOINT}" autoencoder.ckpt
	cd -
fi
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



# Initial tested example for vit_latent with CRPS loss:
# ./scripts/epd.sh 2026/adm 04_vit_latent advection_diffusion_multichannel \
#     encoder@model.encoder=dc_deep \
# 	  decoder@model.decoder=dc_deep \
#     processor@model.processor=vit_latent \
#     model.processor.n_noise_channels=1000 \
#     +model.n_members=10 \
#     model.loss_func._target_=autocast.losses.ensemble.CRPSLoss \
#     +model.train_metrics.crps._target_=autocast.metrics.ensemble.CRPS \
#     logging.wandb.enabled=false \
# 	  model.freeze_encoder_decoder=true \
# 	  model.train_in_latent_space=false \
#     optimizer.learning_rate=0.0002 \
#     trainer.max_epochs=5
