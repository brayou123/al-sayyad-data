#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
from copernicusmarine import subset

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
          depth_max=DEPTH_SURFACE):

    print(f"Fetching {variables}...")

    result = subset(
        dataset_id=dataset_id,
        variables=variables,

        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,

        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,

        start_datetime=TARGET_DATE,
        end_datetime=TARGET_DATE,

        minimum_depth=depth_min,
        maximum_depth=depth_max,

        username=USERNAME,
        password=PASSWORD,

        output_filename=filename
    )

    path = get_path(result)
    return xr.open_dataset(path, engine="netcdf4")

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

    if data.ndim==4:
        profile=data[0,:,ilat,ilon]
    else:
        profile=data[:,ilat,ilon]

    return ds["depth"].values, profile

print("\nDownloading datasets...\n")

ds_sst = fetch(
    "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
    ["analysed_sst"],
    "sst.nc"
)

ds_chl = fetch(
    "cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
    ["CHL"],
    "chl.nc"
)

ds_o2 = fetch(
    "cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    ["o2"],
    "o2.nc"
)

ds_kd = fetch(
    "cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    ["kd490"],
    "kd.nc"
)

ds_cur = fetch(
    "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    ["uo","vo"],
    "cur.nc"
)

ds_sal = fetch(
    "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    ["so"],
    "sal.nc"
)

ds_prof = fetch(
    "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    ["thetao"],
    "prof.nc",
    DEPTH_SURFACE,
    DEPTH_MAX
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

print(f"Temperature : {sst:.2f} °C")
print(f"Chlorophyll : {chl:.3f} mg/m³")
print(f"Oxygen      : {o2:.2f} ml/L")
print(f"Salinity    : {sal:.2f} ppt")
print(f"Secchi      : {secchi:.1f} m")
print(f"Current     : {current_kn:.2f} kn")
print(f"Direction   : {current_dir:.1f}°")
print(f"Thermocline : {thermo_depth:.1f} m")
print(f"Gradient    : {thermo_gradient:.2f} °C")

print("\nTemperature profile:\n")

for d,t in zip(depths,temp):
    if not np.isnan(t):
        print(f"{d:6.1f} m : {t:.2f} °C")

print("\nDone.")