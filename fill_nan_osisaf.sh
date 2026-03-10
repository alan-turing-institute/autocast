#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

# Activate conda environment
source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import torch
import os
from pathlib import Path

base_path = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all")

print("=" * 100)
print("FILLING NaN VALUES WITH 0 IN osisaf_nh_sic_all DATASET")
print("=" * 100)

splits = ["train", "valid", "test"]

for split in splits:
    split_path = base_path / split / "data.pt"
    
    print(f"\n{'=' * 100}")
    print(f"Processing: {split}")
    print(f"{'=' * 100}")
    print(f"Path: {split_path}")
    
    # Load the data
    print(f"Loading data...")
    data_dict = torch.load(split_path, map_location='cpu')
    data_tensor = data_dict["data"]
    
    print(f"Shape: {data_tensor.shape}")
    print(f"Dtype: {data_tensor.dtype}")
    
    # Check NaN statistics
    num_nans_before = torch.isnan(data_tensor).sum().item()
    total_elements = data_tensor.numel()
    nan_percentage_before = (num_nans_before / total_elements) * 100
    
    print(f"\nBefore filling:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  NaN count: {num_nans_before:,}")
    print(f"  NaN percentage: {nan_percentage_before:.4f}%")
    if num_nans_before > 0:
        print(f"  Min (ignoring NaN): {data_tensor[~torch.isnan(data_tensor)].min():.6f}")
        print(f"  Max (ignoring NaN): {data_tensor[~torch.isnan(data_tensor)].max():.6f}")
        print(f"  Mean (ignoring NaN): {data_tensor[~torch.isnan(data_tensor)].mean():.6f}")
    
    # Fill NaN with 0
    print(f"\nFilling NaN values with 0...")
    data_tensor = torch.nan_to_num(data_tensor, nan=0.0)
    data_dict["data"] = data_tensor
    
    # Check after filling
    num_nans_after = torch.isnan(data_tensor).sum().item()
    print(f"\nAfter filling:")
    print(f"  NaN count: {num_nans_after}")
    print(f"  Min: {data_tensor.min():.6f}")
    print(f"  Max: {data_tensor.max():.6f}")
    print(f"  Mean: {data_tensor.mean():.6f}")
    
    # Save back
    print(f"\nSaving modified data...")
    torch.save(data_dict, split_path)
    print(f"✓ Saved to {split_path}")
    
    # Verify the save
    print(f"Verifying save...")
    verify_dict = torch.load(split_path, map_location='cpu')
    verify_tensor = verify_dict["data"]
    verify_nans = torch.isnan(verify_tensor).sum().item()
    print(f"✓ Verification: NaN count after reload = {verify_nans}")

print(f"\n\n{'=' * 100}")
print("✓ COMPLETED: All NaN values in osisaf_nh_sic_all have been replaced with 0")
print(f"{'=' * 100}")

EOF
