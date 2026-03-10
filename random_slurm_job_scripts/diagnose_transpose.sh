#!/bin/bash
#SBATCH --job-name=diagnose_transpose
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00

module purge
module load baskerville
module load bask-apps/live

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python << 'PYTHON_EOF'
import torch
import xarray as xr
import numpy as np
from einops import rearrange

print("=" * 80)
print("DIAGNOSING TRANSPOSE ISSUE")
print("=" * 80)

# 1. Load netCDF and get spatial dimensions
print("\n1. Loading netCDF to understand dimension convention...")
nc_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2018.nc"
ds = xr.open_dataset(nc_path)
sic = ds['ice_conc']
print(f"   NetCDF dims: {sic.dims}")
print(f"   NetCDF shape: {sic.shape}")

# Get one valid timestep (not all zeros)
for i in range(min(10, sic.shape[0])):
    sample = sic.isel(time=i).values
    if not np.all(sample == 0) and not np.all(np.isnan(sample)):
        first_valid = sample
        print(f"   Using timestep {i} for analysis")
        break

print(f"   Sample shape: {first_valid.shape}")
print(f"   Sample has NaN: {np.any(np.isnan(first_valid))}")
print(f"   Sample value range: {np.nanmin(first_valid):.3f} to {np.nanmax(first_valid):.3f}")

# 2. Load mask
print("\n2. Loading mask...")
mask_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt"
mask = torch.load(mask_path).numpy()
print(f"   Mask shape: {mask.shape}")
print(f"   Mask values: {np.unique(mask)}")

# 3. Compare corners - if mask matches data, corners should align
print("\n3. Comparing corner patterns...")
print("   Data (yc, xc) top-left 5x5:")
print(first_valid[:5, :5])
print("\n   Mask (?, ?) top-left 5x5:")
print(mask[:5, :5])
print("\n   Mask TRANSPOSED (.T) top-left 5x5:")
print(mask.T[:5, :5])

# Find a distinctive pattern in data to match
print("\n4. Finding distinctive ocean/land boundary...")
# Look for transition from ocean (valid data) to land (NaN or 0)
# Check middle row
mid_row_idx = mask.shape[0] // 2
data_mid_row = first_valid[mid_row_idx, :]
mask_mid_row = mask[mid_row_idx, :]
mask_mid_col = mask[:, mid_row_idx]  # if transposed

print(f"   Data middle row (yc={mid_row_idx}): valid count = {np.sum(~np.isnan(data_mid_row))}/{len(data_mid_row)}")
print(f"   Mask middle row: ocean count = {np.sum(mask_mid_row==1)}/{len(mask_mid_row)}")
print(f"   Mask middle col: ocean count = {np.sum(mask_mid_col==1)}/{len(mask_mid_col)}")

# 5. Check if data NaN pattern matches mask
print("\n5. Checking NaN/mask alignment...")
is_valid_data = ~np.isnan(first_valid)  # True where data is valid (ocean)
is_ocean_mask = mask == 1  # True where mask says ocean

matches_direct = np.sum(is_valid_data == is_ocean_mask)
total_pixels = mask.size
matches_transposed = np.sum(is_valid_data == is_ocean_mask.T)

print(f"   Pixels matching (mask as-is):     {matches_direct}/{total_pixels} ({100*matches_direct/total_pixels:.1f}%)")
print(f"   Pixels matching (mask transposed): {matches_transposed}/{total_pixels} ({100*matches_transposed/total_pixels:.1f}%)")

# 6. Check rearrange behavior
print("\n6. Testing einops rearrange...")
mask_torch = torch.from_numpy(mask)
mask_rearranged = rearrange(mask_torch, 'w h -> 1 1 w h 1')
print(f"   Original mask shape: {mask_torch.shape}")
print(f"   After rearrange('w h -> 1 1 w h 1'): {mask_rearranged.shape}")
print(f"   Interpretation: (batch, time, w, h, channels) = {mask_rearranged.shape}")

# If we have (yc, xc) and want (batch, time, yc, xc, channels), rearrange is correct
# But if dataset.py expects (batch, time, WIDTH, HEIGHT, channels) and
# unpacks as (width, height), we need to know which is which!

print("\n" + "=" * 80)
print("DIAGNOSIS SUMMARY")
print("=" * 80)

if matches_transposed > matches_direct:
    print("❌ TRANSPOSE MISMATCH CONFIRMED!")
    print(f"   Mask needs .T to match data spatial layout")
    print(f"   Match rate improves from {100*matches_direct/total_pixels:.1f}% to {100*matches_transposed/total_pixels:.1f}%")
    print("\n🔍 ROOT CAUSE:")
    print("   Option A: Mask was saved with (xc, yc) instead of (yc, xc)")
    print("   Option B: Data processing swapped dimensions during conversion")
    print("   Option C: Dataset.py unpacks (width, height) in wrong order")
else:
    print("✓ No transpose issue detected")
    print(f"   Mask aligns correctly with data ({100*matches_direct/total_pixels:.1f}% match)")

print("=" * 80)

# 7. Detailed dimension tracking
print("\n7. DIMENSION TRACKING:")
print("   NetCDF: (time, yc, xc)")
print("   → get_osisaf_data.py: ensures (time, yc, xc) via ensure_order()")
print("   → Adds channel: (time, yc, xc) → (time, yc, xc, 1)")
print("   → Stacks years: (N_years, 365, yc, xc, 1)")
print("   → Dataset unpacks to:", end="")
print("     (n_traj, n_time, self.width, self.height, n_channels)")
print(f"     = (n_traj, n_time, {mask.shape[0]}, {mask.shape[1]}, 1)")
print(f"   So: width={mask.shape[0]} (yc dimension), height={mask.shape[1]} (xc dimension)")
print(f"\n   Mask shape: {mask.shape} - interpreted as (dim0, dim1)")
print(f"   rearrange('w h -> ...') treats:")
print(f"     w = mask.shape[0] = {mask.shape[0]}")
print(f"     h = mask.shape[1] = {mask.shape[1]}")
print(f"\n   If data has (yc, xc) and mask has (yc, xc): ✓ MATCH")
print(f"   If data has (yc, xc) and mask has (xc, yc): ❌ MISMATCH")

PYTHON_EOF

