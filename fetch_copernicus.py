#!/usr/bin/env python3
"""
Al-Sayyad Ocean Data Fetcher
Produces two JSON files:
  - data.json      : Algerian coast + 60km offshore, 1km resolution
  - data_zone2.json: High-resolution zone (300m OLCI), fixed 100km radius
"""

import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from copernicusmarine import subset

# =====================================================
# Credentials
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# Region 1 — Algerian coast + 60km offshore
# =====================================================
R1_LAT_MIN = 36.0
R1_LAT_MAX = 37.5
R1_LON_MIN = -2.0
R1_LON_MAX =  9.0

# =====================================================
# Region 2 — High-resolution fixed zone (100km radius)
# Center kept generic for privacy
# =====================================================
R2_LAT_MIN = 35.7
R2_LAT_MAX = 37.5
R2_LON_MIN =  0.8
R2_LON_MAX =  3.6

# =====================================================
# Common depth settings
# =====================================================
DEPTH_SURFACE     = 1.0182366371154785
DEPTH_MAX_PROFILE = 100.0

# =====================================================
# Date — yesterday
# =====================================================
yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
print(f"Fetching data for: {yesterday}")

# =====================================================
# Shared download helper
# =====================================================
def get_path(result):
    if isinstance(result, str):
        return result
    if hasattr(result, 'filenames') and result.filenames:
        return result.filenames[0]
    if hasattr(result, 'filename'):
        return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}")

def download(dataset_id, variables, filename,
             lat_min, lat_max, lon_min, lon_max,
             depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE):
    print(f"  Downloading {variables} from {dataset_id} -> {filename}")
    result = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min,  maximum_latitude=lat_max,
        start_datetime=yesterday,  end_datetime=yesterday,
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

# =====================================================
# Regrid helper — bilinear interpolation to target grid
# =====================================================
def regrid(ds, var, target_lat, target_lon):
    data = ds[var].values
    data = np.squeeze(data)

    lat_name = next((n for n in ['latitude','lat'] if n in ds.dims), None)
    lon_name = next((n for n in ['longitude','lon'] if n in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError(f"Cannot find lat/lon dims in {list(ds.dims)}")

    src_lat = ds[lat_name].values
    src_lon = ds[lon_name].values

    if data.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze, got shape {data.shape}")

    interp = RegularGridInterpolator(
        (src_lat, src_lon), data,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    la, lo = np.meshgrid(target_lat, target_lon, indexing='ij')
    pts = np.stack([la.ravel(), lo.ravel()], axis=-1)
    return interp(pts).reshape(len(target_lat), len(target_lon))

# =====================================================
# Thermocline from 3D temperature profile
# =====================================================
def compute_thermocline(ds_prof):
    depth_vals = ds_prof.depth.values
    temp_4d    = ds_prof.thetao.values
    temp_3d    = temp_4d[0]

    grad       = np.abs(np.diff(temp_3d, axis=0))
    thermo_idx = np.argmax(grad, axis=0)

    d_upper = depth_vals[:-1]
    d_lower = depth_vals[1:]
    thermo  = (d_upper[thermo_idx] + d_lower[thermo_idx]) / 2.0

    all_nan = np.all(np.isnan(temp_3d), axis=0)
    return np.where(all_nan, np.nan, thermo)

# =====================================================
# FTLE proxy — normalized gradient magnitude of speed
# Used as front/barrier intensity indicator [0..1]
# =====================================================
def compute_ftle_proxy(u2d, v2d, grid_deg=0.01):
    speed = np.sqrt(u2d**2 + v2d**2)
    dy_m  = grid_deg * 111_000
    gx    = np.gradient(speed, axis=1) / dy_m
    gy    = np.gradient(speed, axis=0) / dy_m
    ftle  = np.sqrt(gx**2 + gy**2)
    fmax  = np.nanpercentile(ftle, 98)
    if fmax > 0:
        ftle = np.clip(ftle / fmax, 0, 1)
    return ftle

# =====================================================
# Regrid thermocline map to target grid
# =====================================================
def regrid_thermocline(thermo_2d, src_lat, src_lon, tgt_lat, tgt_lon):
    interp = RegularGridInterpolator(
        (src_lat, src_lon), thermo_2d,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    la, lo = np.meshgrid(tgt_lat, tgt_lon, indexing='ij')
    pts = np.stack([la.ravel(), lo.ravel()], axis=-1)
    return interp(pts).reshape(len(tgt_lat), len(tgt_lon))

# =====================================================
# Build columnar JSON from 2D arrays
# =====================================================
def build_json(lats, lons, sst, chl, sal, o2, kd490,
               u, v, thermo, ftle, resolution_km):
    lat_list, lon_list = [], []
    temp_list, sal_list, chl_list = [], [], []
    o2_list, kd_list, curr_list   = [], [], []
    thermo_list, front_list       = [], []

    for i in range(len(lats)):
        for j in range(len(lons)):
            t = sst[i, j]
            if np.isnan(t):
                continue

            lat_list.append(round(float(lats[i]), 5))
            lon_list.append(round(float(lons[j]), 5))
            temp_list.append(round(float(t), 2))

            s = sal[i, j]
            sal_list.append(round(float(s), 2) if not np.isnan(s) else None)

            c = chl[i, j]
            chl_list.append(round(float(c), 4) if not np.isnan(c) else None)

            o = o2[i, j]
            o2_list.append(round(float(o) / 44.661, 2) if not np.isnan(o) else None)

            k = kd490[i, j]
            if not np.isnan(k) and k > 0.01:
                kd_list.append(round(1.7 / float(k), 1))
            else:
                kd_list.append(None)

            ui, vi = u[i, j], v[i, j]
            if not np.isnan(ui) and not np.isnan(vi):
                curr_list.append(round(float(np.sqrt(ui**2 + vi**2)) * 1.944, 2))
            else:
                curr_list.append(None)

            th = thermo[i, j]
            thermo_list.append(round(float(th), 1) if not np.isnan(th) else None)

            fi = ftle[i, j]
            front_list.append(round(float(fi), 3) if not np.isnan(fi) else 0.5)

    return {
        "timestamp":      yesterday,
        "resolution_km":  resolution_km,
        "lat":            lat_list,
        "lon":            lon_list,
        "temperature":    temp_list,
        "salinity":       sal_list,
        "chlorophyll":    chl_list,
        "oxygen":         o2_list,
        "transparency":   kd_list,
        "currentSpeed":   curr_list,
        "thermocline":    thermo_list,
        "frontIntensity": front_list,
    }

# =====================================================
# Helper — get lat/lon dim names from dataset
# =====================================================
def get_latlon_names(ds):
    lat = next((n for n in ['latitude','lat'] if n in ds.dims), None)
    lon = next((n for n in ['longitude','lon'] if n in ds.dims), None)
    return lat, lon

# ======================================================
# REGION 1 — Algerian coast + 60km offshore
# ======================================================
print("\n=== REGION 1: Algerian coast + 60km offshore ===")

R1 = dict(lat_min=R1_LAT_MIN, lat_max=R1_LAT_MAX,
          lon_min=R1_LON_MIN, lon_max=R1_LON_MAX)

ds_sst1  = download("SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
                    ["analysed_sst"], "r1_sst.nc", **R1)
ds_chl1  = download("cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
                    ["CHL"], "r1_chl.nc", **R1)
ds_o2_1  = download("cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
                    ["o2"], "r1_o2.nc", **R1)
ds_kd1   = download("cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
                    ["kd490"], "r1_kd.nc", **R1)
ds_cur1  = download("cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
                    ["uo","vo"], "r1_cur.nc", **R1)
ds_sal1  = download("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
                    ["so"], "r1_sal.nc", **R1)
ds_prof1 = download("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
                    ["thetao"], "r1_prof.nc",
                    depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX_PROFILE, **R1)

# Target grid 0.01° (~1km)
r1_lats = np.arange(R1_LAT_MIN, R1_LAT_MAX + 0.01, 0.01)
r1_lons = np.arange(R1_LON_MIN, R1_LON_MAX + 0.01, 0.01)
print(f"Grid size: {len(r1_lats)} x {len(r1_lons)} = {len(r1_lats)*len(r1_lons):,} pts")

print("Regriding Region 1...")
sst1  = regrid(ds_sst1, "analysed_sst", r1_lats, r1_lons)
if np.nanmean(sst1) > 100:
    sst1 = sst1 - 273.15
chl1  = regrid(ds_chl1,  "CHL",   r1_lats, r1_lons)
o2_1  = regrid(ds_o2_1,  "o2",    r1_lats, r1_lons)
kd1   = regrid(ds_kd1,   "kd490", r1_lats, r1_lons)
u1    = regrid(ds_cur1,  "uo",    r1_lats, r1_lons)
v1    = regrid(ds_cur1,  "vo",    r1_lats, r1_lons)
sal1  = regrid(ds_sal1,  "so",    r1_lats, r1_lons)

lat_n, lon_n = get_latlon_names(ds_prof1)
thermo1_raw = compute_thermocline(ds_prof1)
thermo1 = regrid_thermocline(thermo1_raw,
                              ds_prof1[lat_n].values,
                              ds_prof1[lon_n].values,
                              r1_lats, r1_lons)
ftle1 = compute_ftle_proxy(u1, v1, grid_deg=0.01)

out1 = build_json(r1_lats, r1_lons, sst1, chl1, sal1,
                  o2_1, kd1, u1, v1, thermo1, ftle1, resolution_km=1.0)

print(f"Region 1: {len(out1['lat']):,} sea points")
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(out1, f, separators=(',', ':'), ensure_ascii=False)
print("Saved data.json")

# ======================================================
# REGION 2 — High-resolution fixed zone
# ======================================================
print("\n=== REGION 2: High-resolution fixed zone ===")

R2 = dict(lat_min=R2_LAT_MIN, lat_max=R2_LAT_MAX,
          lon_min=R2_LON_MIN, lon_max=R2_LON_MAX)

ds_sst2  = download("SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
                    ["analysed_sst"], "r2_sst.nc", **R2)
ds_olci  = download("cmems_obs-oc_med_bgc-plankton_nrt_l3-olci-300m_P1D",
                    ["CHL"], "r2_olci.nc", **R2)
ds_o2_2  = download("cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
                    ["o2"], "r2_o2.nc", **R2)
ds_kd2   = download("cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
                    ["kd490"], "r2_kd.nc", **R2)
ds_cur2  = download("cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
                    ["uo","vo"], "r2_cur.nc", **R2)
ds_sal2  = download("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
                    ["so"], "r2_sal.nc", **R2)
ds_prof2 = download("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
                    ["thetao"], "r2_prof.nc",
                    depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX_PROFILE, **R2)

# Target grid 0.003° (~330m, closest to 300m)
STEP = 0.003
r2_lats = np.arange(R2_LAT_MIN, R2_LAT_MAX + STEP, STEP)
r2_lons = np.arange(R2_LON_MIN, R2_LON_MAX + STEP, STEP)
print(f"Grid size: {len(r2_lats)} x {len(r2_lons)} = {len(r2_lats)*len(r2_lons):,} pts")

print("Regriding Region 2...")
sst2  = regrid(ds_sst2, "analysed_sst", r2_lats, r2_lons)
if np.nanmean(sst2) > 100:
    sst2 = sst2 - 273.15
chl2  = regrid(ds_olci, "CHL",   r2_lats, r2_lons)
o2_2  = regrid(ds_o2_2, "o2",    r2_lats, r2_lons)
kd2   = regrid(ds_kd2,  "kd490", r2_lats, r2_lons)
u2    = regrid(ds_cur2, "uo",    r2_lats, r2_lons)
v2    = regrid(ds_cur2, "vo",    r2_lats, r2_lons)
sal2  = regrid(ds_sal2, "so",    r2_lats, r2_lons)

lat_n2, lon_n2 = get_latlon_names(ds_prof2)
thermo2_raw = compute_thermocline(ds_prof2)
thermo2 = regrid_thermocline(thermo2_raw,
                              ds_prof2[lat_n2].values,
                              ds_prof2[lon_n2].values,
                              r2_lats, r2_lons)
ftle2 = compute_ftle_proxy(u2, v2, grid_deg=STEP)

out2 = build_json(r2_lats, r2_lons, sst2, chl2, sal2,
                  o2_2, kd2, u2, v2, thermo2, ftle2, resolution_km=0.3)

print(f"Region 2: {len(out2['lat']):,} sea points")
with open("data_zone2.json", "w", encoding="utf-8") as f:
    json.dump(out2, f, separators=(',', ':'), ensure_ascii=False)
print("Saved data_zone2.json")

# ======================================================
# Summary
# ======================================================
print("\n=== DONE ===")
print(f"data.json       : {len(out1['lat']):,} pts | 1.0km | Algerian coast")
print(f"data_zone2.json : {len(out2['lat']):,} pts | 0.3km | High-res zone")
print(f"Date            : {yesterday}")
