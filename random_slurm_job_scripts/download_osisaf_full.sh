#!/bin/bash
#SBATCH --job-name=download_osisaf_full
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos=turing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

module purge
module load baskerville
module load bask-apps/live

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

cd /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/code/autocast

python << 'PYTHON_EOF'
import xarray as xr
import os
from pathlib import Path

# Configuration
OPENDAP_URL = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"
RAW_DIR = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf")

# Create output directory
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("DOWNLOADING OSI-SAF FULL DATASET (YEAR BY YEAR)")
print("=" * 80)

# Open the dataset to see available years
print(f"\n1. Connecting to OPeNDAP: {OPENDAP_URL}")
ds = xr.open_dataset(OPENDAP_URL, engine='netcdf4')

print(f"\n2. Dataset info:")
print(f"   Variables: {list(ds.data_vars)}")
print(f"   Coordinates: {list(ds.coords)}")
print(f"   Time range: {ds.time.min().values} to {ds.time.max().values}")

# Get available years
years = sorted(set(ds.time.dt.year.values))
print(f"\n3. Available years: {years[0]} to {years[-1]} ({len(years)} years)")

# Download each year separately
print(f"\n4. Downloading year by year...")
for year in years:
    output_file = RAW_DIR / f"osisaf_nh_{year}.nc"
    
    # Skip if already exists
    if output_file.exists():
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   [{year}] Already exists ({size_mb:.1f} MB), skipping...")
        continue
    
    print(f"   [{year}] Downloading...")
    try:
        # Select this year's data
        ds_year = ds.sel(time=str(year))
        
        # Check if year has data
        n_timesteps = len(ds_year.time)
        if n_timesteps == 0:
            print(f"   [{year}] No data available, skipping...")
            continue
        
        print(f"   [{year}] Found {n_timesteps} timesteps, saving...")
        
        # Save to netCDF
        ds_year.to_netcdf(output_file, engine='netcdf4')
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"   [{year}] ✓ Saved ({size_mb:.1f} MB)")
        
    except Exception as e:
        print(f"   [{year}] ✗ Error: {e}")
        # Remove partial file if it exists
        if output_file.exists():
            output_file.unlink()
        continue

print("\n" + "=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Raw data saved to: {RAW_DIR}")

# List downloaded files
print("\nDownloaded files:")
for f in sorted(RAW_DIR.glob("osisaf_nh_*.nc")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}: {size_mb:.1f} MB")

print("=" * 80)

PYTHON_EOF

