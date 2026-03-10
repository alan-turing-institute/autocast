"""Download individual OSI-SAF yearly files and combine them."""

import xarray as xr
import logging
from pathlib import Path
from urllib.request import urlopen
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

output_dir = Path("/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# List of OSI-SAF files by year (example - you may need to adjust URLs)
base_url = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/reprocessed/ice/"

log.info("Downloading OSI-SAF yearly files...")
log.info("Note: This attempts to download individual files. If THREDDS is down, consider alternative sources.")

# Try alternative: download just ice_conc variable to smaller file
opendap_url = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"

log.info(f"Attempting selective load from {opendap_url}")
try:
    # Load ONLY the ice_conc variable (skip uncertainties)
    ds = xr.open_dataset(opendap_url, engine='netcdf4')
    ice_conc = ds[['ice_conc', 'time', 'lat', 'lon']].copy()
    
    log.info(f"Loaded ice_conc variable: {ice_conc['ice_conc'].shape}")
    log.info(f"Size: {ice_conc.nbytes / 1e9:.2f} GB")
    
    output_file = output_dir / "raw_osisaf_nh_sic_all.nc"
    log.info(f"Saving to {output_file}...")
    
    ice_conc.to_netcdf(output_file, engine='netcdf4', unlimited_dims=['time'])
    
    log.info(f"✓ Success! File saved: {output_file}")
    log.info(f"Size: {output_file.stat().st_size / 1e9:.2f} GB")
    
except Exception as e:
    log.error(f"OPeNDAP failed: {e}")
    log.info("THREDDS server appears to be unstable. Consider:")
    log.info("1. Waiting and retrying later")
    log.info("2. Downloading individual year files from: https://www.osi-saf.org/")
    log.info("3. Using NSIDC or other mirror")
    raise