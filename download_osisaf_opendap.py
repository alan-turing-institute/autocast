#!/usr/bin/env python3
"""Download OSI-SAF data via OPeNDAP."""

import xarray as xr
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# OPeNDAP endpoint
opendap_url = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"

output_path = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw/raw_osisaf_nh_sic_all.nc")
output_path.parent.mkdir(parents=True, exist_ok=True)

log.info(f"Opening OPeNDAP dataset: {opendap_url}")
try:
    # Open with xarray via OPeNDAP
    log.info("Connecting to server...")
    ds = xr.open_dataset(opendap_url, engine='netcdf4')
    
    log.info(f"Dataset loaded successfully!")
    log.info(f"Dimensions: {dict(ds.dims)}")
    log.info(f"Variables: {list(ds.data_vars)}")
    log.info(f"Coordinates: {list(ds.coords)}")
    
    # Get size estimate
    total_size = 0
    for var in ds.data_vars:
        var_size = ds[var].nbytes / 1e9
        log.info(f"  {var}: {var_size:.2f} GB")
        total_size += var_size
    
    log.info(f"Total estimated size: {total_size:.2f} GB")
    
    # Save to netCDF
    log.info(f"\nSaving to {output_path}...")
    ds.to_netcdf(output_path, engine='netcdf4')
    
    final_size = output_path.stat().st_size / 1e9
    log.info(f"✓ Download complete!")
    log.info(f"File size: {final_size:.2f} GB")
    log.info(f"Location: {output_path}")
    
except Exception as e:
    log.error(f"Failed to download: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    raise
