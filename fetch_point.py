#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
from copernicusmarine import subset
from datetime import datetime, timedelta

# Import the specific exception for date-out-of-bounds
from copernicusmarine.core_functions.exceptions import CoordinatesOutOfDatasetBounds

USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials — set env vars")

# ============================================================
# POINT TO ANALYZE
# ============================================================

TARGET_LAT  = 36.580417
TARGET_LON  = 1.870517
TARGET_DATE = "2026-07-11"

# Maximum days to look back if the requested date is unavailable
MAX_LOOKBACK = 7

# تحميل مساحة صغيرة حول النقطة
MARGIN = 0.05

LAT_MIN = TARGET_LAT - MARGIN
LAT_MAX = TARGET_LAT + MARGIN
LON_MIN = TARGET_LON - MARGIN
LON_MAX = TARGET_LON + MARGIN

DEPTH_SURFACE = 1.0182366371154785
DEPTH_MAX     = 100.0

print(f"Target point : {TARGET_LAT}°N, {TARGET_LON}°E")
print(f"Date         : {TARGET_DATE}")
print(f"Max lookback : {MAX_LOOKBACK} days\n")

# Convert target date to datetime for easy subtraction
target_dt = datetime.strptime(TARGET_DATE, "%Y-%m-%d")

def get_path(result):
    if isinstance(result, str):
        return result
    if hasattr(result, "filenames") and result.filenames:
        return result.filenames[0]
    if hasattr(result, "filename"):
        return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}")

def fetch(dataset_id, variables, filename,
          depth_min=DEPTH_SURFACE,
          depth_max=DEPTH_SURFACE,
          date=target_dt):
    """
    Attempt to fetch a dataset for a specific date.
    Raises CoordinatesOutOfDatasetBounds if the date is not available.
    """
    date_str = date.strftime("%Y-%m-%d")
    print(f"  Trying {variables} for {date_str}...")
    result = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=date_str,
        end_datetime=date_str,
        minimum_depth=depth_min,
        maximum_depth=depth_max,
        username=USERNAME,
        password=PASSWORD,
        output_filename=filename
    )
    path = get_path(result)
    ds = xr.open_dataset(path, engine="netcdf4")
    return ds

def fetch_with_retry(dataset_id, variables, filename,
                     depth_min, depth_max,
                     target_dt, max_lookback):
    """
    Try target_dt, then target_dt-1, ... up to max_lookback days.
    Catches ONLY CoordinatesOutOfDatasetBounds to decide on retry.
    All other exceptions are propagated immediately.
    Returns (xarray.Dataset, actual_date, offset_days).
    If no date within max_lookback is available, raises RuntimeError.
    """
    for offset in range(max_lookback + 1):
        date = target_dt - timedelta(days=offset)
        try:
            ds = fetch(dataset_id, variables, filename,
                       depth_min, depth_max, date)
            print(f"  Success with date: {date.strftime('%Y-%m-%d')}")
            return ds, date, offset
        except CoordinatesOutOfDatasetBounds as e:
            print(f"    Date {date.strftime('%Y-%m-%d')} out of bounds, trying earlier...")
            # If this is the last attempt, raise an error
            if offset == max_lookback:
                raise RuntimeError(f"Could not fetch {variables} within {max_lookback} days")
            continue
        except Exception as e:
            # Any other exception (auth, network, server, etc.) must stop immediately
            print(f"    Unexpected error: {e}")
            raise
    # Should never reach here
    raise RuntimeError(f"Unexpected failure for {variables}")

def nearest_value(ds, var):
    data = np.squeeze(ds[var].values)
    lat_name = next(n for n in ["latitude","lat"] if n in ds.coords)
    lon_name = next(n for n in ["longitude","lon"] if n in ds.coords)
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    ilat = np.argmin(np.abs(lats - TARGET_LAT))
    ilon = np.argmin(np.abs(lons - TARGET_LON))
    if data.ndim == 2:
        value = data[ilat, ilon]
    elif data.ndim == 3:
        value = data[0, ilat, ilon]
    elif data.ndim == 4:
        value = data[0,0,ilat,ilon]
    else:
        value = data
    return float(value)

def nearest_profile(ds,var):
    lat_name = next(n for n in ["latitude","lat"] if n in ds.coords)
    lon_name = next(n for n in ["longitude","lon"] if n in ds.coords)
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    ilat = np.argmin(np.abs(lats-TARGET_LAT))
    ilon = np.argmin(np.abs(lons-TARGET_LON))
    data = ds[var].values
    if data.ndim == 4:
        profile=data[0,:,ilat,ilon]
    else:
        profile=data[:,ilat,ilon]
    return ds["depth"].values, profile

print("\nDownloading datasets...\n")

# Download each dataset with the new retry logic
ds_sst, date_sst, offset_sst = fetch_with_retry(
    "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
    ["analysed_sst"],
    "sst.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_chl, date_chl, offset_chl = fetch_with_retry(
    "cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
    ["CHL"],
    "chl.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_o2, date_o2, offset_o2 = fetch_with_retry(
    "cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    ["o2"],
    "o2.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_kd, date_kd, offset_kd = fetch_with_retry(
    "cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    ["kd490"],
    "kd.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_cur, date_cur, offset_cur = fetch_with_retry(
    "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    ["uo","vo"],
    "cur.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_sal, date_sal, offset_sal = fetch_with_retry(
    "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    ["so"],
    "sal.nc",
    DEPTH_SURFACE,
    DEPTH_SURFACE,
    target_dt,
    MAX_LOOKBACK
)

ds_prof, date_prof, offset_prof = fetch_with_retry(
    "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    ["thetao"],
    "prof.nc",
    DEPTH_SURFACE,
    DEPTH_MAX,
    target_dt,
    MAX_LOOKBACK
)

print("\nExtracting values...\n")

sst = nearest_value(ds_sst,"analysed_sst")
if sst>100:
    sst-=273.15

chl = nearest_value(ds_chl,"CHL")
o2 = nearest_value(ds_o2,"o2")/44.661
kd = nearest_value(ds_kd,"kd490")
secchi = 1.7/kd
uo = nearest_value(ds_cur,"uo")
vo = nearest_value(ds_cur,"vo")
current_ms=np.sqrt(uo**2+vo**2)
current_kn=current_ms*1.944
current_dir=(np.degrees(np.arctan2(uo,vo))+360)%360
sal=nearest_value(ds_sal,"so")
depths,temp=nearest_profile(ds_prof,"thetao")
grad=np.abs(np.diff(temp))
grad[:2]=0
idx=np.argmax(grad)
thermo_depth=(depths[idx]+depths[idx+1])/2
thermo_gradient=grad[idx]

print("\n===============================")
print("POINT DATA")
print("===============================")

# Helper to format offset relative to TARGET_DATE
def format_offset(offset):
    if offset == 0:
        return "0 days"
    elif offset == 1:
        return "-1 day"
    else:
        return f"-{offset} days"

print(f"Temperature : {sst:.2f} °C ({date_prof.strftime('%Y-%m-%d')}, {format_offset(offset_prof)})")
print(f"Chlorophyll : {chl:.3f} mg/m³ ({date_chl.strftime('%Y-%m-%d')}, {format_offset(offset_chl)})")
print(f"Oxygen      : {o2:.2f} ml/L ({date_o2.strftime('%Y-%m-%d')}, {format_offset(offset_o2)})")
print(f"Salinity    : {sal:.2f} ppt ({date_sal.strftime('%Y-%m-%d')}, {format_offset(offset_sal)})")
print(f"Secchi      : {secchi:.1f} m ({date_kd.strftime('%Y-%m-%d')}, {format_offset(offset_kd)})")
print(f"Current     : {current_kn:.2f} kn ({date_cur.strftime('%Y-%m-%d')}, {format_offset(offset_cur)})")
print(f"Direction   : {current_dir:.1f}°")
print(f"Thermocline : {thermo_depth:.1f} m")
print(f"Gradient    : {thermo_gradient:.2f} °C")

print("\nTemperature profile:\n")
for d,t in zip(depths,temp):
    if not np.isnan(t):
        print(f"{d:6.1f} m : {t:.2f} °C")

# Print warnings for datasets older than 2 days
print("\n===============================")
print("WARNINGS (data > 2 days old)")
print("===============================")
if offset_sst > 2:
    print(f"WARNING: SST data is {offset_sst} days older than the requested date.")
if offset_chl > 2:
    print(f"WARNING: Chlorophyll data is {offset_chl} days older than the requested date.")
if offset_o2 > 2:
    print(f"WARNING: Oxygen data is {offset_o2} days older than the requested date.")
if offset_kd > 2:
    print(f"WARNING: Secchi (kd490) data is {offset_kd} days older than the requested date.")
if offset_cur > 2:
    print(f"WARNING: Current data is {offset_cur} days older than the requested date.")
if offset_sal > 2:
    print(f"WARNING: Salinity data is {offset_sal} days older than the requested date.")
if offset_prof > 2:
    print(f"WARNING: Temperature profile data is {offset_prof} days older than the requested date.")

# Final summary of actual dates used
print("\n===============================")
print("ACTUAL DATES USED")
print("===============================")
print(f"SST          : {date_sst.strftime('%Y-%m-%d')} (offset: {offset_sst} days)")
print(f"CHL          : {date_chl.strftime('%Y-%m-%d')} (offset: {offset_chl} days)")
print(f"Oxygen       : {date_o2.strftime('%Y-%m-%d')} (offset: {offset_o2} days)")
print(f"Current      : {date_cur.strftime('%Y-%m-%d')} (offset: {offset_cur} days)")
print(f"Salinity     : {date_sal.strftime('%Y-%m-%d')} (offset: {offset_sal} days)")
print(f"Temperature  : {date_prof.strftime('%Y-%m-%d')} (offset: {offset_prof} days)")

print("\nDone.")