#!/bin/bash
#SBATCH --job-name=check_nc_dims
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
import xarray as xr
import torch
import numpy as np

print("=" * 80)
print("CHECKING NETCDF DIMENSIONS vs MASK DIMENSIONS")
print("=" * 80)

# Load netCDF file
nc_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf/osisaf_nh_2018.nc"
print(f"\n1. Loading netCDF: {nc_path}")
ds = xr.open_dataset(nc_path)

print(f"\nDataset variables: {list(ds.data_vars)}")
print(f"Dataset coords: {list(ds.coords)}")
print(f"Dataset dims: {dict(ds.dims)}")

# Find SIC variable
sic_var = None
for var_name in ['sic', 'ice_conc', 'sea_ice_concentration', 'concentration']:
    if var_name in ds.data_vars:
        sic_var = var_name
        break
if sic_var is None:
    sic_var = list(ds.data_vars)[0]

print(f"\n2. Using SIC variable: '{sic_var}'")
sic = ds[sic_var]
print(f"   Shape: {sic.shape}")
print(f"   Dims: {sic.dims}")
print(f"   Dtype: {sic.dtype}")

# Get one timestep to see spatial dimensions
print(f"\n3. Extracting first timestep...")
first_timestep = sic.isel(time=0).values
print(f"   Shape of first timestep: {first_timestep.shape}")
print(f"   First 3x3 corner:")
print(first_timestep[:3, :3])

# Load mask
print(f"\n4. Loading land mask...")
mask_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic_all/land_mask.pt"
mask = torch.load(mask_path)
print(f"   Mask shape: {mask.shape}")
print(f"   First 3x3 corner of mask:")
print(mask[:3, :3])

# Load processed data (if exists)
print(f"\n5. Loading processed PyTorch data...")
try:
    data_path = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/processed_autocast/osisaf_nh_sic/data.pt"
    data_dict = torch.load(data_path)
    if isinstance(data_dict, dict) and 'data' in data_dict:
        data = data_dict['data']
    else:
        data = data_dict
    print(f"   Data shape: {data.shape}")
    print(f"   Dims interpretation: (traj, time, spatial_0, spatial_1, channels)")
    
    # Get first trajectory, first timestep
    first_sample = data[0, 0, :, :, 0]
    print(f"   First sample shape: {first_sample.shape}")
    print(f"   First 3x3 corner of processed data:")
    print(first_sample[:3, :3])
    
except Exception as e:
    print(f"   Could not load: {e}")

# Check dimension order in netCDF
print(f"\n6. Analyzing dimension order...")
spatial_dims = [d for d in sic.dims if d != 'time']
print(f"   Spatial dimensions (in order): {spatial_dims}")

try:
    dim0_coords = ds.coords[spatial_dims[0]]
    dim1_coords = ds.coords[spatial_dims[1]]
    print(f"   {spatial_dims[0]} range: {dim0_coords.min().values} to {dim0_coords.max().values}, size={dim0_coords.size}")
    print(f"   {spatial_dims[1]} range: {dim1_coords.min().values} to {dim1_coords.max().values}, size={dim1_coords.size}")
except:
    print("   Could not get coordinate ranges")

print("\n" + "=" * 80)
print("DIMENSION ANALYSIS SUMMARY")
print("=" * 80)
print(f"NetCDF dimensions: {sic.dims}")
print(f"NetCDF spatial order: {' → '.join(spatial_dims)}")
print(f"Mask shape: {mask.shape}")
print(f"\nPOTENTIAL ISSUE:")
print(f"  If netCDF uses (time, yc, xc) or (time, lat, lon)")
print(f"  But mask was created with different (row, col) convention")
print(f"  Then mask needs transpose to match data spatial layout!")
print("=" * 80)

PYTHON_EOF

