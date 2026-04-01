#!/usr/bin/env python3
"""
Advanced Mediterranean Sea Operational System
- Uses working model products (4.2km) that are reliable
- Lagrangian Coherent Structures (FTLE)
- Master Index (Biological Explosion Index)
- Ocean Memory (Bayesian Recursive Filter)
- Output: compact binary file (.bio)
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset
from scipy.interpolate import RegularGridInterpolator
import struct

# =====================================================
# Credentials
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# Coordinates (using the dataset's actual bounds to avoid warnings)
# =====================================================
LAT_MIN = 30.1875          # Minimum latitude of the model product
LAT_MAX = 38.5             # Algerian coast + 60km (still within dataset)
LON_MIN = -5.541666507720947
LON_MAX = 36.29166793823242
DEPTH_SURFACE = 1.0182366371154785
DEPTH_MIN_PROFILE = DEPTH_SURFACE
DEPTH_MAX_PROFILE = 100.0

# Grid resolution for final output (0.01° ≈ 1.1 km)
GRID_STEP_DEG = 0.01

# =====================================================
# Time
# =====================================================
yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# Helper functions
# =====================================================
def get_path_from_result(result):
    if isinstance(result, str):
        return result
    if hasattr(result, 'filenames') and result.filenames:
        return result.filenames[0]
    if hasattr(result, 'filename'):
        return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}: {result}")

def download_and_open(dataset_id, variables, filename, depth_min=None, depth_max=None):
    print(f"Downloading {variables} from {dataset_id}...")
    result = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=yesterday, end_datetime=yesterday,
        minimum_depth=depth_min, maximum_depth=depth_max,
        username=USERNAME, password=PASSWORD,
        output_filename=filename
    )
    path = get_path_from_result(result)
    if not os.path.exists(path):
        raise RuntimeError(f"File {path} not created")
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise RuntimeError(f"File {path} is empty (size 0)")
    print(f"  Downloaded {path} ({file_size} bytes)")
    return xr.open_dataset(path, engine='netcdf4')

def regrid_to_target(ds, var_name, target_lat, target_lon):
    """
    Regrid a 2D field to a regular lat/lon grid using bilinear interpolation.
    Works with data that may have extra dimensions (time, depth) of size 1.
    """
    # Get original grid coordinates
    if 'latitude' in ds.dims:
        orig_lat = ds.latitude.values
        orig_lon = ds.longitude.values
    elif 'lat' in ds.dims:
        orig_lat = ds.lat.values
        orig_lon = ds.lon.values
    else:
        raise ValueError("Cannot find latitude/longitude dimensions")
    
    # Get data and squeeze to remove any dimensions of size 1
    data = ds[var_name].values
    data = np.squeeze(data)   # remove any singleton dimensions (time, depth, etc.)
    
    # After squeeze, we expect a 2D array (lat, lon)
    if data.ndim != 2:
        raise ValueError(f"Data has {data.ndim} dimensions after squeeze; expected 2")
    
    # Create interpolator
    interp = RegularGridInterpolator(
        (orig_lat, orig_lon), data,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    # Generate target grid points
    lat_mesh, lon_mesh = np.meshgrid(target_lat, target_lon, indexing='ij')
    points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
    regridded = interp(points).reshape(target_lat.shape[0], target_lon.shape[0])
    return regridded

def compute_ftle(u, v, dx_deg, dy_deg):
    """Simplified FTLE proxy using gradient magnitude."""
    lat_rad = np.deg2rad(np.linspace(LAT_MIN, LAT_MAX, u.shape[0]))
    dy_m = dy_deg * 111e3
    dx_m = dx_deg * 111e3 * np.cos(lat_rad)[:, np.newaxis]
    speed = np.sqrt(u**2 + v**2)
    grad_speed_x = np.gradient(speed, axis=1) / dx_m
    grad_speed_y = np.gradient(speed, axis=0) / dy_m
    ftle = np.sqrt(grad_speed_x**2 + grad_speed_y**2)
    return ftle

def compute_master_index(ftle, chl, thermocline, vorticity):
    """Master Index combining FTLE, chlorophyll, thermocline, vorticity."""
    ftle_norm = np.clip(ftle / 0.5, 0, 1)
    chl_logistic = 1 / (1 + np.exp(-10 * (chl - 0.2)))
    thermo_factor = np.sin(np.pi * thermocline / 100)
    vort_norm = np.clip(np.abs(vorticity) / 1e-5, 0, 0.5)
    hotspot = ftle_norm * chl_logistic * (0.5 + 0.5 * thermo_factor) * (1 + vort_norm)
    return np.clip(hotspot, 0, 1)

def update_ocean_memory(prev_memory, current_hotspot, u, v):
    """Bayesian update: advect previous memory with currents, then blend."""
    if prev_memory is None:
        return current_hotspot
    mean_u = np.nanmean(u)
    mean_v = np.nanmean(v)
    dt_hours = 24
    dx_km = mean_u * dt_hours * 3600 / 1000
    dy_km = mean_v * dt_hours * 3600 / 1000
    dx_cell = dx_km / (GRID_STEP_DEG * 111)
    dy_cell = dy_km / (GRID_STEP_DEG * 111)
    shift_x = int(round(dx_cell))
    shift_y = int(round(dy_cell))
    advected = np.roll(prev_memory, shift=(shift_y, shift_x), axis=(0, 1))
    new_memory = 0.7 * advected + 0.3 * current_hotspot
    return new_memory

def save_binary(output_path, header, layers):
    """Save multiple 2D arrays as binary with header."""
    with open(output_path, 'wb') as f:
        f.write(struct.pack('I', header['version']))
        f.write(struct.pack('I', header['nlat']))
        f.write(struct.pack('I', header['nlon']))
        f.write(struct.pack('f', header['lat_min']))
        f.write(struct.pack('f', header['lat_max']))
        f.write(struct.pack('f', header['lon_min']))
        f.write(struct.pack('f', header['lon_max']))
        f.write(struct.pack('f', header['step_deg']))
        f.write(struct.pack('I', len(layers)))
        for name, data in layers.items():
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            data_flat = data.astype(np.float32).flatten()
            f.write(data_flat.tobytes())

# =====================================================
# Main processing
# =====================================================
def main():
    # 1. Define target grid
    lat_full = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_DEG, GRID_STEP_DEG)
    lon_full = np.arange(LON_MIN, LON_MAX + GRID_STEP_DEG, GRID_STEP_DEG)
    nlat, nlon = len(lat_full), len(lon_full)

    # 2. Download all products (only surface depth for 2D fields, full depth for profile)
    # Temperature (surface)
    ds_temp = download_and_open(
        "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        ["thetao"],
        "temp.nc",
        depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE
    )
    # Salinity (surface)
    ds_sal = download_and_open(
        "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
        ["so"],
        "sal.nc",
        depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE
    )
    # Currents (surface)
    ds_cur = download_and_open(
        "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        ["uo", "vo"],
        "cur.nc",
        depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE
    )
    # Chlorophyll (surface)
    ds_chl = download_and_open(
        "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
        ["chl"],
        "chl.nc",
        depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE
    )
    # Temperature profile (full depth 0-100m)
    ds_prof = download_and_open(
        "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        ["thetao"],
        "profile.nc",
        depth_min=DEPTH_MIN_PROFILE, depth_max=DEPTH_MAX_PROFILE
    )

    # 3. Regrid to target grid (0.01°)
    print("Regridding to common grid...")
    sst_reg = regrid_to_target(ds_temp, "thetao", lat_full, lon_full)
    chl_reg = regrid_to_target(ds_chl, "chl", lat_full, lon_full)
    u_reg = regrid_to_target(ds_cur, "uo", lat_full, lon_full)
    v_reg = regrid_to_target(ds_cur, "vo", lat_full, lon_full)
    sal_reg = regrid_to_target(ds_sal, "so", lat_full, lon_full)

    # 4. Thermocline from 3D profile
    # Get 3D temperature (depth, lat, lon)
    temp_3d = ds_prof.thetao.values
    # Squeeze to remove time if present
    temp_3d = np.squeeze(temp_3d)
    if temp_3d.ndim == 3:
        # Assuming dimensions: (depth, lat, lon)
        depth_vals = ds_prof.depth.values
        grad = np.abs(np.diff(temp_3d, axis=0))
        thermo_idx = np.argmax(grad, axis=0)
        depth_upper = depth_vals[:-1]
        depth_lower = depth_vals[1:]
        thermo_raw = (depth_upper[thermo_idx] + depth_lower[thermo_idx]) / 2
    else:
        raise ValueError("Profile data is not 3D after squeezing")

    # Regrid thermocline (nearest neighbor)
    orig_lat = ds_prof.latitude.values
    orig_lon = ds_prof.longitude.values
    lat_idx = np.argmin(np.abs(orig_lat[:, np.newaxis] - lat_full), axis=0)
    lon_idx = np.argmin(np.abs(orig_lon[:, np.newaxis] - lon_full), axis=0)
    thermo_reg = thermo_raw[np.ix_(lat_idx, lon_idx)]

    # 5. Compute FTLE
    ftle_map = compute_ftle(u_reg, v_reg, GRID_STEP_DEG, GRID_STEP_DEG)

    # 6. Predict chlorophyll drift 12h ahead
    mean_u = np.nanmean(u_reg)
    mean_v = np.nanmean(v_reg)
    dx_km = mean_u * 12 * 3600 / 1000
    dy_km = mean_v * 12 * 3600 / 1000
    dx_cell = dx_km / (GRID_STEP_DEG * 111)
    dy_cell = dy_km / (GRID_STEP_DEG * 111)
    from scipy.ndimage import shift
    chl_pred = shift(chl_reg, (dy_cell, dx_cell), order=1, mode='nearest')

    # 7. Compute vorticity
    dx_m = GRID_STEP_DEG * 111e3 * np.cos(np.deg2rad(lat_full))[:, np.newaxis]
    dy_m = GRID_STEP_DEG * 111e3
    dv_dx = np.gradient(v_reg, axis=1) / dx_m
    du_dy = np.gradient(u_reg, axis=0) / dy_m
    vorticity = dv_dx - du_dy

    # 8. Master Index
    hotspot = compute_master_index(ftle_map, chl_pred, thermo_reg, vorticity)

    # 9. Ocean Memory
    memory_file = "ocean_memory.npy"
    if os.path.exists(memory_file):
        prev_memory = np.load(memory_file)
    else:
        prev_memory = None
    ocean_memory = update_ocean_memory(prev_memory, hotspot, u_reg, v_reg)
    np.save(memory_file, ocean_memory)

    # 10. Assemble layers for binary output
    current_speed = np.sqrt(u_reg**2 + v_reg**2) * 1.944  # knots
    layers = {
        "hotspot": hotspot,
        "ftle": ftle_map,
        "sst": sst_reg,
        "chl_pred": chl_pred,
        "current_speed": current_speed,
        "thermocline": thermo_reg,
        "salinity": sal_reg,
        "ocean_memory": ocean_memory,
    }

    # 11. Save binary file
    header = {
        'version': 1,
        'nlat': nlat,
        'nlon': nlon,
        'lat_min': LAT_MIN,
        'lat_max': LAT_MAX,
        'lon_min': LON_MIN,
        'lon_max': LON_MAX,
        'step_deg': GRID_STEP_DEG,
    }
    save_binary("data.bio", header, layers)
    print(f"Saved data.bio with {len(layers)} layers")

if __name__ == "__main__":
    main()