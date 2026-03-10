#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name fill_nan_osisaf
#SBATCH --output=logs/fill_nan_osisaf_%j.out
#SBATCH --error=logs/fill_nan_osisaf_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
import os

print("Filling NaN values with 0 in osisaf_nh_sic_all dataset\n")

base_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all"

for split in ["train", "valid", "test"]:
    split_path = os.path.join(base_path, split, "data.pt")
    
    print(f"Processing {split}...")
    
    # Load
    data_dict = torch.load(split_path)
    data_tensor = data_dict["data"]
    
    # Count NaN before
    nan_before = torch.isnan(data_tensor).sum().item()
    
    # Fill NaN with 0
    data_tensor[torch.isnan(data_tensor)] = 0.0
    
    # Count NaN after
    nan_after = torch.isnan(data_tensor).sum().item()
    
    # Save
    torch.save(data_dict, split_path)
    
    print(f"  Shape: {data_tensor.shape}")
    print(f"  NaN before: {nan_before:,}")
    print(f"  NaN after: {nan_after:,}")
    print(f"  ✓ Saved\n")

print("Done!")

EOF
