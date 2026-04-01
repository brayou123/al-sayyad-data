#!/usr/bin/env python3
"""
Advanced Mediterranean Sea Operational System
- Daily data retrieval from Copernicus Marine
- Lagrangian Coherent Structures (FTLE)
- Master Index (Biological Explosion Index)
- Ocean Memory (Bayesian Recursive Filter)
- High-resolution (300m) enhancement for user-defined region
- Output: compact binary file (.bio)
"""

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset
import scipy.ndimage as ndimage
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
# Coordinates (you can change these)
# =====================================================
LAT_MIN = 30.0          # Southern boundary
LAT_MAX = 38.5          # Northern boundary (Algerian coast + 60km)
LON_MIN = -3.0          # Western boundary
LON_MAX = 10.0          # Eastern boundary
DEPTH_SURFACE = 1.0182366371154785
DEPTH_MIN_PROFILE = DEPTH_SURFACE
DEPTH_MAX_PROFILE = 100.0

# Grid resolution for full coverage (0.01° ≈ 1.1 km)
GRID_STEP_DEG = 0.01

# High-resolution region (e.g., Cherchell 100km radius)
HR_LAT_CENTER = float(os.environ.get("HR_LAT_CENTER", 36.6))
HR_LON_CENTER = float(os.environ.get("HR_LON_CENTER", 2.2))
HR_RADIUS_KM = float(os.environ.get("HR_RADIUS_KM", 100.0))

# =====================================================
# Time
# =====================================================
yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# Helper functions (same as before)
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

def download_high_res(bbox, filename):
    """Download high-resolution (300m) OLCI product for a small bounding box."""
    print(f"Downloading high-res (300m) for region: {bbox}")
    try:
        result = subset(
            dataset_id="cmems_obs-oc_med_bgc-plankton_nrt_l3-olci-300m_P1D",
            variables=["CHL"],
            minimum_longitude=bbox[0], maximum_longitude=bbox[1],
            minimum_latitude=bbox[2], maximum_latitude=bbox[3],
            start_datetime=yesterday, end_datetime=yesterday,
            username=USERNAME, password=PASSWORD,
            output_filename=filename
        )
        path = get_path_from_result(result)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"High-res file {path} missing or empty")
        print(f"  Downloaded high-res: {path} ({os.path.getsize(path)} bytes)")
        return xr.open_dataset(path, engine='netcdf4')
    except Exception as e:
        print(f"  High-res download failed: {e}")
        return None

def regrid_to_target(ds, var_name, target_lat, target_lon):
    """Regrid a 2D field to a regular lat/lon grid using bilinear interpolation."""
    # Get original grid
    if 'latitude' in ds.dims:
        orig_lat = ds.latitude.values
        orig_lon = ds.longitude.values
    elif 'lat' in ds.dims:
        orig_lat = ds.lat.values
        orig_lon = ds.lon.values
    else:
        raise ValueError("Cannot find latitude/longitude dimensions")
    
    data = ds[var_name].values
    # If time dimension exists, take first time step
    if data.ndim == 3:
        data = data[0]
    
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
# High-resolution processing
# =====================================================
def process_high_res(lat_full, lon_full, chl_full):
    """
    Download and integrate high-resolution (300m) OLCI data for user-defined region.
    Returns an enhancement layer (2D array same shape as full grid) or None if failed.
    """
    if HR_RADIUS_KM <= 0:
        return None

    # Compute bounding box around center
    lat_hr_min = HR_LAT_CENTER - (HR_RADIUS_KM / 111.0)
    lat_hr_max = HR_LAT_CENTER + (HR_RADIUS_KM / 111.0)
    lon_hr_min = HR_LON_CENTER - (HR_RADIUS_KM / (111.0 * np.cos(np.deg2rad(HR_LAT_CENTER))))
    lon_hr_max = HR_LON_CENTER + (HR_RADIUS_KM / (111.0 * np.cos(np.deg2rad(HR_LAT_CENTER))))

    # Ensure bounds within full domain
    lon_hr_min = max(LON_MIN, lon_hr_min)
    lon_hr_max = min(LON_MAX, lon_hr_max)
    lat_hr_min = max(LAT_MIN, lat_hr_min)
    lat_hr_max = min(LAT_MAX, lat_hr_max)

    if lon_hr_min >= lon_hr_max or lat_hr_min >= lat_hr_max:
        print("High-res region is outside full domain. Skipping.")
        return None

    bbox = (lon_hr_min, lon_hr_max, lat_hr_min, lat_hr_max)
    ds_hr = download_high_res(bbox, "chl_300m.nc")
    if ds_hr is None:
        print("High-res data not available for today.")
        return None

    # Determine variable name (usually 'CHL')
    chl_var = "CHL" if "CHL" in ds_hr.variables else None
    if chl_var is None:
        for var in ds_hr.variables:
            if "CHL" in var or "chl" in var:
                chl_var = var
                break
    if chl_var is None:
        print("No chlorophyll variable found in high-res file.")
        return None

    # Extract high-res data (2D)
    hr_chl = ds_hr[chl_var].values
    if hr_chl.ndim == 3:
        hr_chl = hr_chl[0]  # first time
    # Get original high-res grid
    hr_lat = ds_hr.latitude.values if 'latitude' in ds_hr.dims else ds_hr.lat.values
    hr_lon = ds_hr.longitude.values if 'longitude' in ds_hr.dims else ds_hr.lon.values

    # Create a mask for the high-res region in the full grid
    mask = (lat_full >= lat_hr_min) & (lat_full <= lat_hr_max)
    mask_lon = (lon_full >= lon_hr_min) & (lon_full <= lon_hr_max)
    # Indices
    lat_idx = np.where(mask)[0]
    lon_idx = np.where(mask_lon)[0]
    if len(lat_idx) == 0 or len(lon_idx) == 0:
        return None

    # Regrid high-res data to the full grid subregion using bilinear interpolation
    # Create a fine target grid for the subregion (same resolution as full grid)
    sub_lat = lat_full[lat_idx[0]:lat_idx[-1]+1]
    sub_lon = lon_full[lon_idx[0]:lon_idx[-1]+1]
    # Build interpolator for high-res data
    interp_hr = RegularGridInterpolator(
        (hr_lat, hr_lon), hr_chl,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    lat_mesh, lon_mesh = np.meshgrid(sub_lat, sub_lon, indexing='ij')
    points = np.stack([lat_mesh.ravel(), lon_mesh.ravel()], axis=-1)
    hr_regridded = interp_hr(points).reshape(len(sub_lat), len(sub_lon))

    # Replace NaN in high-res with values from the full chlorophyll field
    full_sub = chl_full[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    nan_mask = np.isnan(hr_regridded)
    hr_regridded[nan_mask] = full_sub[nan_mask]

    # Create enhancement layer: ratio of high-res to full (or difference)
    # We'll compute a factor to boost hotspot in that region
    # Simple approach: enhancement = 1 + (hr_regridded / full_sub) * 0.5, capped.
    # But to avoid division by zero, use full_sub as denominator with small epsilon.
    epsilon = 1e-6
    enhancement_factor = 1 + 0.5 * (hr_regridded / (full_sub + epsilon))
    enhancement_factor = np.clip(enhancement_factor, 1, 2)

    # Build full-sized enhancement array (default 1)
    enhancement = np.ones_like(chl_full)
    enhancement[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1] = enhancement_factor

    return enhancement

# =====================================================
# Main processing
# =====================================================
def main():
    # 1. Define target grid
    lat_full = np.arange(LAT_MIN, LAT_MAX + GRID_STEP_DEG, GRID_STEP_DEG)
    lon_full = np.arange(LON_MIN, LON_MAX + GRID_STEP_DEG, GRID_STEP_DEG)
    nlat, nlon = len(lat_full), len(lon_full)

    # 2. Download all products
    # SST L4 (1km)
    ds_sst = download_and_open(
    "SST_MED_SST_L4_NRT_OBSERVATIONS_010_004",
    ["analysed_sst"],
    "sst.nc"
)

    # Ocean colour plankton (1km multi-sensor)
    ds_chl = download_and_open(
        "cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
        ["CHL"],
        "chl.nc"
    )
    # Fallback: rename if variable name differs
    if "CHL" not in ds_chl.variables:
        for var in ds_chl.variables:
            if "CHL" in var or "chl" in var:
                ds_chl = ds_chl.rename({var: "CHL"})
                break

    # Currents (4.2km)
    ds_cur = download_and_open(
        "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
        ["uo", "vo"],
        "cur.nc"
    )
    # Salinity (4.2km)
    ds_sal = download_and_open(
        "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
        ["so"],
        "sal.nc"
    )
    # Temperature profile (4.2km)
    ds_prof = download_and_open(
        "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
        ["thetao"],
        "profile.nc",
        depth_min=DEPTH_MIN_PROFILE, depth_max=DEPTH_MAX_PROFILE
    )

    # 3. Regrid to target grid (0.01°)
    print("Regridding to common grid...")
    sst_reg = regrid_to_target(ds_sst, "analysed_sst", lat_full, lon_full)
    chl_reg = regrid_to_target(ds_chl, "CHL", lat_full, lon_full)
    u_reg = regrid_to_target(ds_cur, "uo", lat_full, lon_full)
    v_reg = regrid_to_target(ds_cur, "vo", lat_full, lon_full)
    sal_reg = regrid_to_target(ds_sal, "so", lat_full, lon_full)

    # 4. Thermocline from 3D profile
    depth_vals = ds_prof.depth.values
    temp_4d = ds_prof.thetao.values
    temp_3d = temp_4d[0]
    grad = np.abs(np.diff(temp_3d, axis=0))
    thermo_idx = np.argmax(grad, axis=0)
    depth_upper = depth_vals[:-1]
    depth_lower = depth_vals[1:]
    thermo_raw = (depth_upper[thermo_idx] + depth_lower[thermo_idx]) / 2
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

    # 8. Master Index (without high-res)
    hotspot = compute_master_index(ftle_map, chl_pred, thermo_reg, vorticity)

    # 9. Ocean Memory
    memory_file = "ocean_memory.npy"
    if os.path.exists(memory_file):
        prev_memory = np.load(memory_file)
    else:
        prev_memory = None
    ocean_memory = update_ocean_memory(prev_memory, hotspot, u_reg, v_reg)
    np.save(memory_file, ocean_memory)

    # 10. High-resolution enhancement (300m)
    high_res_enhancement = process_high_res(lat_full, lon_full, chl_reg)
    if high_res_enhancement is not None:
        # Apply enhancement to hotspot (optional: you can also keep separate layer)
        hotspot_enhanced = hotspot * high_res_enhancement
        hotspot_enhanced = np.clip(hotspot_enhanced, 0, 1)
        # Use the enhanced version as the main hotspot
        hotspot = hotspot_enhanced

    # 11. Assemble layers for binary output
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
    if high_res_enhancement is not None:
        layers["high_res_enhancement"] = high_res_enhancement

    # 12. Save binary file
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