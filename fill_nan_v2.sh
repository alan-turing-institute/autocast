#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=0
#SBATCH --mem=64G
#SBATCH --job-name fill_nan_v2
#SBATCH --output=logs/fill_nan_v2_%j.out
#SBATCH --error=logs/fill_nan_v2_%j.err

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

print("=" * 100)
print("FILLING NaN VALUES WITH 0 IN osisaf_nh_sic_all DATASET")
print("=" * 100)

base_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all"

splits = ["train", "valid", "test"]

for split in splits:
    split_path = os.path.join(base_path, split, "data.pt")
    
    print(f"\n{'=' * 100}")
    print(f"Processing: {split}")
    print(f"Path: {split_path}")
    print(f"{'=' * 100}")
    
    # Load the data
    print("Loading data...")
    data_dict = torch.load(split_path)
    data_tensor = data_dict["data"]
    
    print(f"Shape: {data_tensor.shape}")
    print(f"Dtype: {data_tensor.dtype}")
    
    # Count NaN before
    nan_count_before = torch.isnan(data_tensor).sum().item()
    total_elements = data_tensor.numel()
    nan_pct_before = (nan_count_before / total_elements) * 100
    
    print(f"\nBefore filling:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  NaN count: {nan_count_before:,}")
    print(f"  NaN percentage: {nan_pct_before:.4f}%")
    
    # Fill NaN with 0 IN PLACE
    print("\nFilling NaN with 0...")
    data_tensor[torch.isnan(data_tensor)] = 0.0
    
    # Count NaN after
    nan_count_after = torch.isnan(data_tensor).sum().item()
    nan_pct_after = (nan_count_after / total_elements) * 100
    
    print(f"\nAfter filling:")
    print(f"  NaN count: {nan_count_after:,}")
    print(f"  NaN percentage: {nan_pct_after:.4f}%")
    
    # Show value statistics (only if not all zeros)
    non_zero = (data_tensor != 0).sum().item()
    print(f"  Non-zero elements: {non_zero:,}")
    
    if non_zero > 0:
        min_val = data_tensor[data_tensor != 0].min().item()
        max_val = data_tensor[data_tensor != 0].max().item()
        mean_val = data_tensor[data_tensor != 0].mean().item()
        print(f"  Value range (non-zero): [{min_val:.6f}, {max_val:.6f}]")
        print(f"  Mean (non-zero): {mean_val:.6f}")
    
    # Save back
    print(f"\nSaving back to {split_path}...")
    torch.save(data_dict, split_path)
    
    # Verify by reloading
    print("Verifying by reloading...")
    verify_dict = torch.load(split_path)
    verify_tensor = verify_dict["data"]
    verify_nan_count = torch.isnan(verify_tensor).sum().item()
    
    if verify_nan_count == 0:
        print("✓ Verification PASSED - No NaN values in saved file")
    else:
        print(f"✗ Verification FAILED - Still has {verify_nan_count} NaN values!")
        raise RuntimeError(f"Verification failed for {split}")

print("\n" + "=" * 100)
print("COMPLETED SUCCESSFULLY")
print("=" * 100)
print("All NaN values in osisaf_nh_sic_all dataset have been filled with 0")

EOF
