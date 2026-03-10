#!/bin/bash
#SBATCH --account=vjgo8416-ai-phy-sys
#SBATCH --qos turing
#SBATCH --time 03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --job-name download_osisaf
#SBATCH --output=logs/download_osisaf_%j.out
#SBATCH --error=logs/download_osisaf_%j.err

set -e

module purge
module load baskerville
module load bask-apps/live
module load Python/3.11.3-GCCcore-12.3.0

source /bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/miniconda3/etc/profile.d/conda.sh
conda activate autocast

python << 'EOF'
import xarray as xr
import os
import pandas as pd

print("=" * 100)
print("DOWNLOADING OSI-SAF DATASET BY YEAR")
print("=" * 100)

# OPENDAP URL for aggregated dataset
opendap_url = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"
output_dir = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw_osisaf"

os.makedirs(output_dir, exist_ok=True)

# Years to download (adjust range as needed)
# OSI-450-a1 covers 1979-2015 (CDR)
# For full coverage, you might want 1979-2023 or check available years
start_year = 1978
end_year = 2020

print(f"\nOpening OPENDAP dataset: {opendap_url}")
print(f"Output directory: {output_dir}")
print(f"Downloading years: {start_year} to {end_year}")
print("=" * 100)

try:
    # Open the remote dataset (lazy loading)
    print("\nConnecting to THREDDS server...")
    ds = xr.open_dataset(opendap_url, engine='netcdf4', chunks={'time': 365})
    
    print(f"✓ Connected successfully!")
    print(f"\nDataset info:")
    print(f"  Dimensions: {dict(ds.dims)}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Download year by year
    for year in range(start_year, end_year + 1):
        output_file = os.path.join(output_dir, f"osisaf_nh_{year}.nc")
        
        # Skip if already downloaded
        if os.path.exists(output_file):
            print(f"\n[{year}] File already exists, skipping: {output_file}")
            continue
        
        print(f"\n[{year}] Selecting data for year {year}...")
        
        try:
            # Select year's data
            ds_year = ds.sel(time=str(year))
            
            # Check if year has data
            if len(ds_year.time) == 0:
                print(f"[{year}] ⚠ No data available, skipping")
                continue
            
            print(f"[{year}] Found {len(ds_year.time)} timesteps")
            print(f"[{year}] Downloading...")
            
            # Download and save
            ds_year.to_netcdf(output_file)
            
            file_size = os.path.getsize(output_file) / (1024**2)  # MB
            print(f"[{year}] ✓ Downloaded: {output_file} ({file_size:.1f} MB)")
            
            ds_year.close()
            
        except Exception as e:
            print(f"[{year}] ✗ Error: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            continue
    
    print("\n" + "=" * 100)
    print("DOWNLOAD COMPLETE")
    print("=" * 100)
    
    # List all downloaded files
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.nc')])
    print(f"\nTotal files downloaded: {len(files)}")
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f)) / (1024**2)
        print(f"  {f}: {size:.1f} MB")
    
    ds.close()
    
except Exception as e:
    print(f"\n✗ Fatal error: {e}")
    raise

EOF
