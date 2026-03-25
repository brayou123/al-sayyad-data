#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# =====================================================
# Credentials
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# Coordinates
# =====================================================
LAT_MIN = 30.1875
LAT_MAX = 45.97916793823242
LON_MIN = -5.541666507720947
LON_MAX = 36.29166793823242
DEPTH_SURFACE = 1.0182366371154785
DEPTH_MIN_PROFILE = DEPTH_SURFACE
DEPTH_MAX_PROFILE = 100.0

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

def download_and_open(dataset_id, variables, filename, depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE):
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

# =====================================================
# Download surface datasets
# =====================================================
ds_temp = download_and_open(
    "cmems_mod_med_phy-tem_anfc_4.2km_P1D-m", ["thetao"], "temp.nc"
)
ds_sal = download_and_open(
    "cmems_mod_med_phy-sal_anfc_4.2km_P1D-m", ["so"], "sal.nc"
)
ds_cur = download_and_open(
    "cmems_mod_med_phy-cur_anfc_4.2km_P1D-m", ["uo", "vo"], "cur.nc"
)
ds_chl = download_and_open(
    "cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m", ["chl"], "chl.nc"
)
ds_o2 = download_and_open(
    "cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m", ["o2"], "o2.nc"
)
ds_kd = download_and_open(
    "cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m", ["kd490"], "kd490.nc"
)

# =====================================================
# Download temperature profile
# =====================================================
print("Downloading temperature profile (surface to 100m)...")
profile_result = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_MIN_PROFILE, maximum_depth=DEPTH_MAX_PROFILE,
    username=USERNAME, password=PASSWORD,
    output_filename="profile.nc"
)
profile_path = get_path_from_result(profile_result)
if not os.path.exists(profile_path):
    raise RuntimeError(f"Profile file {profile_path} not created")
print(f"  Downloaded profile.nc ({os.path.getsize(profile_path)} bytes)")
ds_prof = xr.open_dataset(profile_path, engine='netcdf4')

# =====================================================
# Merge surface datasets
# =====================================================
print("Aligning and merging surface datasets...")
datasets = [
    ds_temp.squeeze(),
    ds_sal.squeeze(),
    ds_cur.squeeze(),
    ds_chl.squeeze(),
    ds_o2.squeeze(),
    ds_kd.squeeze()
]
aligned = xr.align(*datasets, join='inner')
ds_surf = xr.merge(aligned, compat='override')
if 'time' in ds_surf.dims:
    ds_surf = ds_surf.isel(time=0, drop=True)
elif 'time' in ds_surf.coords:
    ds_surf = ds_surf.drop_vars('time')

# =====================================================
# Compute thermocline (vectorized)
# =====================================================
print("Computing thermocline...")
depth_vals = ds_prof.depth.values
temp_4d = ds_prof.thetao.values
temp_3d = temp_4d[0]  # first time only
grad = np.abs(np.diff(temp_3d, axis=0))
thermo_idx = np.argmax(grad, axis=0)
depth_upper = depth_vals[:-1]
depth_lower = depth_vals[1:]
thermocline_map = (depth_upper[thermo_idx] + depth_lower[thermo_idx]) / 2
all_nan_mask = np.all(np.isnan(temp_3d), axis=0)
thermocline_map = np.where(all_nan_mask, np.nan, thermocline_map)

# =====================================================
# Build columnar data and compute front intensity efficiently
# =====================================================
print("Building columnar data...")
lats = ds_surf.latitude.values
lons = ds_surf.longitude.values

# Collect all valid temperature points and their indices for front calculation
valid_temp_points = []
valid_temp_lats = []
valid_temp_lons = []
valid_temp_vals = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = float(ds_surf.thetao.isel(latitude=i, longitude=j).values)
        if not np.isnan(temp_val):
            valid_temp_points.append((lat, lon, temp_val, i, j))
            valid_temp_lats.append(lat)
            valid_temp_lons.append(lon)
            valid_temp_vals.append(temp_val)

if not valid_temp_points:
    raise RuntimeError("No valid temperature data for front calculation")

# Create a regular grid with 0.1° step for front calculation (faster)
min_lat = min(valid_temp_lats)
max_lat = max(valid_temp_lats)
min_lon = min(valid_temp_lons)
max_lon = max(valid_temp_lons)
step = 0.1  # degrees (approx 11 km)
lat_grid = np.arange(min_lat, max_lat + step, step)
lon_grid = np.arange(min_lon, max_lon + step, step)
nlat = len(lat_grid)
nlon = len(lon_grid)

print(f"Creating grid {nlat} x {nlon} for front calculation")

# Assign each point to the nearest grid cell (by rounding)
# This avoids expensive distance loops
lat_indices = np.round((valid_temp_lats - min_lat) / step).astype(int)
lon_indices = np.round((valid_temp_lons - min_lon) / step).astype(int)
# Clip indices to valid range
lat_indices = np.clip(lat_indices, 0, nlat-1)
lon_indices = np.clip(lon_indices, 0, nlon-1)

# Build grid temperature: for each cell, take average of points assigned to it
grid_temp = np.full((nlat, nlon), np.nan)
grid_counts = np.zeros((nlat, nlon), dtype=int)
for idx in range(len(valid_temp_points)):
    i = lat_indices[idx]
    j = lon_indices[idx]
    if np.isnan(grid_temp[i, j]):
        grid_temp[i, j] = valid_temp_vals[idx]
    else:
        # average (optional: could keep only nearest, but average is fine)
        grid_temp[i, j] = (grid_temp[i, j] * grid_counts[i, j] + valid_temp_vals[idx]) / (grid_counts[i, j] + 1)
    grid_counts[i, j] += 1

# Sobel operator using numpy (vectorized)
# Compute gradient in x (lon) and y (lat) directions
# Use central difference for interior cells
grad_x = np.zeros((nlat, nlon))
grad_y = np.zeros((nlat, nlon))

# Sobel kernels
for i in range(1, nlat-1):
    for j in range(1, nlon-1):
        if np.isnan(grid_temp[i, j]):
            continue
        # Sobel Gx (horizontal)
        gx = (grid_temp[i-1, j+1] if not np.isnan(grid_temp[i-1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j-1] if not np.isnan(grid_temp[i-1, j-1]) else grid_temp[i, j]) \
             + 2 * ((grid_temp[i, j+1] if not np.isnan(grid_temp[i, j+1]) else grid_temp[i, j]) \
                  - (grid_temp[i, j-1] if not np.isnan(grid_temp[i, j-1]) else grid_temp[i, j])) \
             + (grid_temp[i+1, j+1] if not np.isnan(grid_temp[i+1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i+1, j-1] if not np.isnan(grid_temp[i+1, j-1]) else grid_temp[i, j])
        # Sobel Gy (vertical)
        gy = (grid_temp[i+1, j-1] if not np.isnan(grid_temp[i+1, j-1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j-1] if not np.isnan(grid_temp[i-1, j-1]) else grid_temp[i, j]) \
             + 2 * ((grid_temp[i+1, j] if not np.isnan(grid_temp[i+1, j]) else grid_temp[i, j]) \
                  - (grid_temp[i-1, j] if not np.isnan(grid_temp[i-1, j]) else grid_temp[i, j])) \
             + (grid_temp[i+1, j+1] if not np.isnan(grid_temp[i+1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j+1] if not np.isnan(grid_temp[i-1, j+1]) else grid_temp[i, j])
        # Convert to °C per km
        lat_rad = lat_grid[i] * np.pi / 180
        km_per_deg_lat = 111
        km_per_deg_lon = 111 * np.cos(lat_rad)
        grad_x[i, j] = gx / (2 * step) / km_per_deg_lon
        grad_y[i, j] = gy / (2 * step) / km_per_deg_lat

grad_mag = np.sqrt(grad_x**2 + grad_y**2)

# Now assign front intensity to each original point using the same grid indices
# We already have lat_indices and lon_indices for valid points; we can reuse them
# For all points (including those with NaN temperature, we will assign 0.5 later)
front_intensity_map = np.zeros((nlat, nlon))
for i in range(nlat):
    for j in range(nlon):
        if not np.isnan(grad_mag[i, j]):
            front_intensity_map[i, j] = min(1.0, grad_mag[i, j] / 0.3)
        else:
            front_intensity_map[i, j] = 0.5  # default

# Now iterate over all surface points (including those with NaN temperature) to build output
lat_list = []
lon_list = []
temp_list = []
sal_list = []
chl_list = []
o2_list = []
kd_list = []
curr_list = []
thermo_list = []
front_list = []

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = float(ds_surf.thetao.isel(latitude=i, longitude=j).values)
        if np.isnan(temp_val):
            continue

        lat_list.append(round(float(lat), 6))
        lon_list.append(round(float(lon), 6))
        temp_list.append(round(temp_val, 2))

        sal_val = float(ds_surf.so.isel(latitude=i, longitude=j).values)
        sal_list.append(round(sal_val, 2) if not np.isnan(sal_val) else None)

        u_val = float(ds_surf.uo.isel(latitude=i, longitude=j).values)
        v_val = float(ds_surf.vo.isel(latitude=i, longitude=j).values)
        if not np.isnan(u_val) and not np.isnan(v_val):
            curr_knots = round(np.sqrt(u_val**2 + v_val**2) * 1.944, 2)
        else:
            curr_knots = None
        curr_list.append(curr_knots)

        chl_val = float(ds_surf.chl.isel(latitude=i, longitude=j).values)
        chl_list.append(round(chl_val, 4) if not np.isnan(chl_val) else None)

        o2_val = float(ds_surf.o2.isel(latitude=i, longitude=j).values)
        if not np.isnan(o2_val):
            o2_ml = round(o2_val / 44.661, 2)
        else:
            o2_ml = None
        o2_list.append(o2_ml)

        kd_val = float(ds_surf.kd490.isel(latitude=i, longitude=j).values)
        if not np.isnan(kd_val) and kd_val > 0.01:
            secchi = round(1.7 / kd_val, 1)
        else:
            secchi = None
        kd_list.append(secchi)

        thermo_val = thermocline_map[i, j]
        thermo_list.append(round(float(thermo_val), 1) if not np.isnan(thermo_val) else None)

        # Front intensity: find nearest grid cell using rounding (same as before)
        i_grid = int(round((lat - min_lat) / step))
        j_grid = int(round((lon - min_lon) / step))
        i_grid = np.clip(i_grid, 0, nlat-1)
        j_grid = np.clip(j_grid, 0, nlon-1)
        front_list.append(round(front_intensity_map[i_grid, j_grid], 3))

# =====================================================
# Save data.json
# =====================================================
output = {
    "timestamp": yesterday,
    "resolution_km": 4.2,
    "lat": lat_list,
    "lon": lon_list,
    "temperature": temp_list,
    "salinity": sal_list,
    "chlorophyll": chl_list,
    "oxygen": o2_list,
    "transparency": kd_list,
    "currentSpeed": curr_list,
    "thermocline": thermo_list,
    "frontIntensity": front_list
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print(f"Saved {len(lat_list)} points")
print("data.json saved successfully")
print(f"Sample front intensity: {front_list[:3]}")
