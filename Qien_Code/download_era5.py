"""Download ERA5 single-level reanalysis data for years 2000-2020."""

import logging
from pathlib import Path

import cdsapi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/projects/u6eo/qiencai/seaice/ERA5")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "mean_wave_direction",
    "sea_surface_temperature",
    "significant_height_of_combined_wind_waves_and_swell",
    "surface_pressure",
    "total_precipitation",
    "downward_uv_radiation_at_the_surface",
]

MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

DAYS = [
    "01", "02", "03", "04", "05", "06", "07",
    "08", "09", "10", "11", "12", "13", "14",
    "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25", "26", "27", "28",
    "29", "30", "31",
]

client = cdsapi.Client()

for year in range(2000, 2021):
    output_file = OUTPUT_DIR / f"era5_single_levels_{year}.grib"

    if output_file.exists():
        log.info(f"Already exists, skipping: {output_file}")
        continue

    log.info(f"Downloading ERA5 year {year} -> {output_file}")

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": ["12:00"],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [90, -180, 30, 180],
    }

    try:
        client.retrieve(DATASET, request).download(str(output_file))
        log.info(f"Saved: {output_file} ({output_file.stat().st_size / 1e9:.2f} GB)")
    except Exception as e:
        log.error(f"Failed for year {year}: {e}")

log.info("ERA5 download complete.")
