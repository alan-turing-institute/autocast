# save_osisaf_raw.py
import os
import xarray as xr

OPENDAP_URL = "https://thredds.met.no/thredds/dodsC/osisaf/met.no/reprocessed/ice/conc_450a1_nh_agg"
RAW_DIR = "/bask/projects/v/vjgo8416-ai-phy-sys/qqaa9560/data/seaice/osisaf_raw"

os.makedirs(RAW_DIR, exist_ok=True)

print("Opening OPENDAP (raw)...")
ds = xr.open_dataset(OPENDAP_URL)

out_path = os.path.join(RAW_DIR, "osisaf_nh_raw.nc")
ds.to_netcdf(out_path)

print(f"Saved raw dataset → {out_path}")
