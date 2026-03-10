#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --job-name=create_seaice_mask
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=logs/create_seaice_mask_%j.log
#SBATCH --error=logs/create_seaice_mask_%j.err

# Activate conda environment
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

# Change to script directory
cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the mask creation script
echo "Starting sea ice land mask creation..."
python scripts/create_seaice_mask.py

echo "Mask creation completed!"
