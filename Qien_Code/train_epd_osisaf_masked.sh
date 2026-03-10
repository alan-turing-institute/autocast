#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 4:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --mem=64G
#SBATCH --job-name epd_masked_osisaf

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

# Using selected years data (2014-2020: 5 years train, 1 year valid, 1 year test)
# Previous: experiment_name=seaice/epd_flow_pixels_in2_out1_masked__2018_data
# Previous: datamodule.data_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_2018/osisaf_nh_sic_2018

/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/.conda/envs/autocast/bin/python -m autocast.scripts.train.encoder_processor_decoder \
  experiment_name=seaice/epd_flow_pixels_in2_out1_masked__selectedyears \
  datamodule=osisaf_nh_sic \
  datamodule.data_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_osisaf_selectedyears \
  trainer.max_epochs=40 \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  logging.wandb.enabled=true \
  encoder@model.encoder=identity \
  decoder@model.decoder=identity \
  processor@model.processor=masked_flow_matching \
  model.processor.backbone.global_cond_channels=null \
  model.processor.backbone.include_global_cond=false \
  model.processor.mask_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/land_mask.pt
