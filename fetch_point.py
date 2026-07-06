#!/usr/bin/env python3
"""
Al-Sayyad — Point Data Fetcher
Fetches environmental data for a specific location and date
from Copernicus Marine Service.

Usage: python fetch_point.py
"""

import os
import numpy as np
import xarray as xr
from copernicusmarine import subset

# =====================================================
# Credentials
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials — set env vars")

# =====================================================
# Target point & date
# =====================================================
TARGET_LAT  = 35.9043
TARGET_LON  = -0.1649
TARGET_DATE = "2026-06-03"

# Small bounding box around the point (±0.05° ~ 5km)
MARGIN = 0.05
LAT_MIN = TARGET_LAT - MARGIN
LAT_MAX = TARGET_LAT + MARGIN
LON_MIN = TARGET_LON - MARGIN
LON_MAX = TARGET_LON + MARGIN

DEPTH_SURFACE = 1.0182366371154785
DEPTH_MAX     = 100.0   # fetch down to 100m for thermocline

print(f"Target point : {TARGET_LAT}°N, {TARGET_LON}°E")
print(f"Date         : {TARGET_DATE}")
print(f"Bounding box : lat [{LAT_MIN:.3f}, {LAT_MAX:.3f}] | lon [{LON_MIN:.3f}, {LON_MAX:.3f}]")
print()

# =====================================================
# Helper — download + extract nearest point value
# =====================================================
def get_path(result):
    if isinstance(result, str):
        return result
    if hasattr(result, 'filenames') and result.filenames:
        return result.filenames[0]
    if hasattr(result, 'filename'):
        return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}")

def fetch(dataset_id, variables, filename,
          depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE):
    print(f"  Fetching {variables} ...")
    result = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,  maximum_latitude=LAT_MAX,
        start_datetime=TARGET_DATE, end_datetime=TARGET_DATE,
        minimum_depth=depth_min,   maximum_depth=depth_max,
        username=USERNAME,         password=PASSWORD,
        output_filename=filename
    )
    path = get_path(result)
    size = os.path.getsize(path)
    if size == 0:
        raise RuntimeError(f"{filename} is empty")
    print(f"    OK — {size:,} bytes")
    return xr.open_dataset(path, engine='netcdf4')

def nearest_value(ds, var):
    """Extract value nearest to target lat/lon (surface level)"""
    data = ds[var].values
    data = np.squeeze(data)

    lat_name = next((n for n in ['latitude','lat'] if n in ds.dims), None)
    lon_name = next((n for n in ['longitude','lon'] if n in ds.dims), None)

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    # Find nearest indices
    ilat = np.argmin(np.abs(lats - TARGET_LAT))
    ilon = np.argmin(np.abs(lons - TARGET_LON))

    if data.ndim == 2:
        val = data[ilat, ilon]
    elif data.ndim == 1:
        val = data[ilat]
    else:
        val = data[ilat, ilon]

    return float(val) if not np.isnan(val) else None

def nearest_profile(ds, var):
    """Extract vertical profile nearest to target lat/lon"""
    lat_name = next((n for n in ['latitude','lat'] if n in ds.dims), None)
    lon_name = next((n for n in ['longitude','lon'] if n in ds.dims), None)

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    ilat = np.argmin(np.abs(lats - TARGET_LAT))
    ilon = np.argmin(np.abs(lons - TARGET_LON))

    # data shape: (time, depth, lat, lon) or (depth, lat, lon)
    data = ds[var].values
    if data.ndim == 4:
        profile = data[0, :, ilat, ilon]
    elif data.ndim == 3:
        profile = data[:, ilat, ilon]
    else:
        profile = data

    depths = ds['depth'].values
    return depths, profile

# =====================================================
# Fetch all variables
# =====================================================
print("=== Fetching surface variables ===")

ds_sst = fetch("SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
               ["analysed_sst"], "pt_sst.nc")

ds_chl = fetch("cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
               ["CHL"], "pt_chl.nc")

ds_o2  = fetch("cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
               ["o2"], "pt_o2.nc")

ds_kd  = fetch("cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
               ["kd490"], "pt_kd.nc")

ds_cur = fetch("cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
               ["uo","vo"], "pt_cur.nc")

ds_sal = fetch("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
               ["so"], "pt_sal.nc")

print("\n=== Fetching temperature profile (0–100m) ===")
ds_prof = fetch("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
                ["thetao"], "pt_prof.nc",
                depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX)

# =====================================================
# Extract values
# =====================================================
print("\n=== Extracting values ===")

# SST
sst = nearest_value(ds_sst, "analysed_sst")
if sst and sst > 100:
    sst = sst - 273.15

# CHL
chl = nearest_value(ds_chl, "CHL")

# Oxygen — convert mmol/m³ → ml/L
o2_raw = nearest_value(ds_o2, "o2")
o2 = round(o2_raw / 44.661, 2) if o2_raw else None

# Transparency — Secchi from Kd490
kd = nearest_value(ds_kd, "kd490")
secchi = round(1.7 / kd, 1) if kd and kd > 0.01 else None

# Current speed
uo = nearest_value(ds_cur, "uo")
vo = nearest_value(ds_cur, "vo")
current_ms  = round(np.sqrt(uo**2 + vo**2), 3) if uo and vo else None
current_kn  = round(current_ms * 1.944, 2) if current_ms else None
current_dir = round(np.degrees(np.arctan2(uo, vo)) % 360, 1) if uo and vo else None

# Salinity
sal = nearest_value(ds_sal, "so")

# Thermocline from temperature profile
depths, temp_profile = nearest_profile(ds_prof, "thetao")
valid = ~np.isnan(temp_profile)
thermo_depth = None
thermo_gradient = None

if valid.sum() >= 2:
    t_valid = temp_profile[valid]
    d_valid = depths[valid]
    grad = np.abs(np.diff(t_valid))
    idx  = np.argmax(grad)
    thermo_depth    = round((d_valid[idx] + d_valid[idx+1]) / 2, 1)
    thermo_gradient = round(float(grad[idx]), 2)

# =====================================================
# Print results
# =====================================================
print()
print("=" * 50)
print(f"  POINT DATA — {TARGET_DATE}")
print(f"  {TARGET_LAT}°N, {TARGET_LON}°E — Arzew Bay")
print("=" * 50)
print()
print(f"  🌡️  Temperature (surface) : {round(sst,2) if sst else '—'} °C")
print(f"  🟢  Chlorophyll           : {round(chl,3) if chl else '—'} mg/m³")
print(f"  💧  Oxygen                : {o2 if o2 else '—'} ml/L")
print(f"  🧂  Salinity              : {round(sal,2) if sal else '—'} ppt")
print(f"  👁️  Transparency (Secchi)  : {secchi if secchi else '—'} m")
print(f"  🌊  Current speed         : {current_kn if current_kn else '—'} kn ({current_ms} m/s)")
print(f"  🧭  Current direction     : {current_dir if current_dir else '—'} °")
print(f"  📏  Thermocline depth     : {thermo_depth if thermo_depth else '—'} m")
print(f"  📐  Thermocline gradient  : {thermo_gradient if thermo_gradient else '—'} °C/m")
print()

# Temperature profile
print("  Temperature profile:")
for d, t in zip(depths, temp_profile):
    if not np.isnan(t) and d <= DEPTH_MAX:
        marker = " ◄ THERMOCLINE" if thermo_depth and abs(d - thermo_depth) < 5 else ""
        print(f"    {d:6.1f} m : {t:.2f} °C{marker}")

print()
print("Done.")
