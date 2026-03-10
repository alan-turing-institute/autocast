#!/bin/bash
#SBATCH --job-name=transpose_mask
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00

module purge
module load baskerville
module load bask-apps/live

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python << 'PYTHON_EOF'
import torch
import os
from pathlib import Path

mask_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt"

print("=" * 80)
print("TRANSPOSING LAND MASK")
print("=" * 80)

# Load original mask
print(f"\n1. Loading original mask from: {mask_path}")
mask_original = torch.load(mask_path)
print(f"   Original shape: {mask_original.shape}")
print(f"   Original dtype: {mask_original.dtype}")

# Create backup
backup_path = mask_path.replace(".pt", "_original_backup.pt")
if not os.path.exists(backup_path):
    print(f"\n2. Creating backup at: {backup_path}")
    torch.save(mask_original, backup_path)
    print("   ✓ Backup created")
else:
    print(f"\n2. Backup already exists at: {backup_path}")

# Transpose the mask
print(f"\n3. Transposing mask...")
mask_transposed = mask_original.T
print(f"   Transposed shape: {mask_transposed.shape}")

# Verify stats are preserved
land_pixels_orig = (mask_original == 0).sum().item()
ocean_pixels_orig = (mask_original == 1).sum().item()
land_pixels_trans = (mask_transposed == 0).sum().item()
ocean_pixels_trans = (mask_transposed == 1).sum().item()

print(f"\n4. Verifying statistics...")
print(f"   Original  - Land: {land_pixels_orig:,}, Ocean: {ocean_pixels_orig:,}")
print(f"   Transposed - Land: {land_pixels_trans:,}, Ocean: {ocean_pixels_trans:,}")

if land_pixels_orig == land_pixels_trans and ocean_pixels_orig == ocean_pixels_trans:
    print("   ✓ Statistics match (transpose is correct)")
else:
    print("   ✗ WARNING: Statistics don't match!")

# Save transposed mask with original name
print(f"\n5. Saving transposed mask to: {mask_path}")
torch.save(mask_transposed, mask_path)
print("   ✓ Transposed mask saved")

print("\n" + "=" * 80)
print("MASK TRANSPOSITION COMPLETE")
print("=" * 80)
print(f"Original mask backed up to: {backup_path}")
print(f"Transposed mask saved to: {mask_path}")
print("\nThe model will now use the correctly oriented mask!")
print("=" * 80)

PYTHON_EOF

