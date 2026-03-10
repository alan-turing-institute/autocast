#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 2:00:00
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --tasks-per-node 4
#SBATCH --job-name train_masked_epd

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.10.8-GCCcore-12.2.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast
export PYTHONPATH=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast/src:$PYTHONPATH

python -m autocast.scripts.train.encoder_processor_decoder \
  --config-path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast/configs \
  datamodule=osisaf_nh_sic \
  model.processor=masked_flow_matching \
  +model.processor.mask_path=/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt \
  trainer.max_epochs=2 \
  trainer.accelerator=gpu \
  trainer.devices=1
