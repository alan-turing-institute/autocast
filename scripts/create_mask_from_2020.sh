#!/bin/bash
#SBATCH --job-name=create_mask_2020
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

module purge
module load baskerville
module load bask-apps/live

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python << 'PYTHON_EOF'
import torch
import numpy as np
import xarray as xr
from pathlib import Path

print("=" * 80)
print("CREATING AND TRANSPOSING SEA ICE LAND MASK FROM 2020 DATA")
print("=" * 80)

# Step 1: Create mask from raw netCDF
nc_file = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2020.nc"
output_dir = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf"
mask_path = Path(output_dir) / "land_mask.pt"

# Backup existing mask
if mask_path.exists():
    backup_path = mask_path.with_suffix('.pt.backup')
    print(f"\n1. Backing up existing mask to: {backup_path}")
    import shutil
    shutil.copy(mask_path, backup_path)

print(f"\n2. Creating new mask from: {nc_file}")
print(f"Loading netCDF file...")
ds = xr.open_dataset(nc_file)
print(f"Dataset variables: {list(ds.data_vars)}")

# Try to find the sea ice concentration variable
sic_var = None
for var_name in ['sic', 'ice_conc', 'sea_ice_concentration', 'concentration']:
    if var_name in ds.data_vars:
        sic_var = var_name
        break

if sic_var is None:
    sic_var = list(ds.data_vars)[0]

print(f"Using variable: {sic_var}")
sic_data = ds[sic_var].values
print(f"SIC data shape: {sic_data.shape}")

# Create mask: 1 for ocean (valid data), 0 for land (invalid/NaN)
if sic_data.ndim == 3:
    # Time series data
    valid_per_pixel = np.sum(~np.isnan(sic_data), axis=0)
    total_timesteps = sic_data.shape[0]
    coverage = valid_per_pixel / total_timesteps
    mask = (coverage >= 0.5).astype(np.float32)
    print(f"Created mask from time series (50% coverage threshold)")
else:
    mask = (~np.isnan(sic_data)).astype(np.float32)
    print(f"Created mask from static data")

mask_tensor = torch.from_numpy(mask).float()
print(f"Mask shape: {mask_tensor.shape}")
print(f"Land pixels (0): {(mask_tensor == 0).sum().item():,}")
print(f"Ocean pixels (1): {(mask_tensor == 1).sum().item():,}")

ds.close()

# Step 2: Save mask (no transpose)
print(f"\n3. Saving mask (original orientation)...")
torch.save(mask_tensor, mask_path)
print(f"Saved mask to: {mask_path}")

print("\n" + "=" * 80)
print("MASK CREATION COMPLETE (FROM 2020 DATA - NO TRANSPOSE)")
print("=" * 80)
print(f"Path: {mask_path}")
print(f"Shape: {mask_tensor.shape}")
print(f"Ready to use in training!")
print("=" * 80)

PYTHON_EOF
