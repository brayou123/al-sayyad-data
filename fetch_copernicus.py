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
# Helper to extract path
# =====================================================
def get_path_from_result(result):
    if isinstance(result, str):
        return result
    if hasattr(result, 'filenames') and result.filenames:
        return result.filenames[0]
    if hasattr(result, 'filename'):
        return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}: {result}")

# =====================================================
# Download and open
# =====================================================
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
ds_surf = xr.merge(aligned)
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
# Build columnar data with front intensity (Sobel on regular grid)
# =====================================================
print("Building columnar data...")
lats = ds_surf.latitude.values
lons = ds_surf.longitude.values

# Collect all valid temperature points for front calculation
valid_points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = float(ds_surf.thetao.isel(latitude=i, longitude=j).values)
        if np.isnan(temp_val):
            continue
        valid_points.append((lat, lon, temp_val))

if len(valid_points) == 0:
    raise RuntimeError("No valid temperature data for front calculation")

# Create regular grid for Sobel (step 0.05° ~ 5.5 km)
min_lat = min(p[0] for p in valid_points)
max_lat = max(p[0] for p in valid_points)
min_lon = min(p[1] for p in valid_points)
max_lon = max(p[1] for p in valid_points)
step = 0.05
lat_grid = np.arange(min_lat, max_lat + step, step)
lon_grid = np.arange(min_lon, max_lon + step, step)
nlat = len(lat_grid)
nlon = len(lon_grid)

print(f"Creating grid {nlat} x {nlon} for front calculation")

# Interpolate temperature onto grid using IDW (fast nearest neighbor approximation)
grid_temp = np.full((nlat, nlon), np.nan)
for i, lat in enumerate(lat_grid):
    for j, lon in enumerate(lon_grid):
        # Find points within 0.2 deg (~22 km)
        candidates = [(p[2], haversine(lat, lon, p[0], p[1])) for p in valid_points
                      if abs(p[0] - lat) < 0.2 and abs(p[1] - lon) < 0.2]
        if not candidates:
            continue
        # Simple inverse distance weighting
        weights = [1.0/(d+0.001) for _, d in candidates]
        temp = np.average([t for t, _ in candidates], weights=weights)
        grid_temp[i, j] = temp

# Sobel operator to compute gradient magnitude (°C per km)
grad_mag = np.zeros((nlat, nlon))
for i in range(1, nlat-1):
    for j in range(1, nlon-1):
        if np.isnan(grid_temp[i, j]):
            continue
        # Sobel kernels
        gx = (grid_temp[i-1, j+1] if not np.isnan(grid_temp[i-1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j-1] if not np.isnan(grid_temp[i-1, j-1]) else grid_temp[i, j]) \
             + 2*((grid_temp[i, j+1] if not np.isnan(grid_temp[i, j+1]) else grid_temp[i, j]) \
                 - (grid_temp[i, j-1] if not np.isnan(grid_temp[i, j-1]) else grid_temp[i, j])) \
             + (grid_temp[i+1, j+1] if not np.isnan(grid_temp[i+1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i+1, j-1] if not np.isnan(grid_temp[i+1, j-1]) else grid_temp[i, j])
        gy = (grid_temp[i+1, j-1] if not np.isnan(grid_temp[i+1, j-1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j-1] if not np.isnan(grid_temp[i-1, j-1]) else grid_temp[i, j]) \
             + 2*((grid_temp[i+1, j] if not np.isnan(grid_temp[i+1, j]) else grid_temp[i, j]) \
                 - (grid_temp[i-1, j] if not np.isnan(grid_temp[i-1, j]) else grid_temp[i, j])) \
             + (grid_temp[i+1, j+1] if not np.isnan(grid_temp[i+1, j+1]) else grid_temp[i, j]) \
             - (grid_temp[i-1, j+1] if not np.isnan(grid_temp[i-1, j+1]) else grid_temp[i, j])
        # Convert to °C per km: 1 deg latitude ≈ 111 km, 1 deg longitude ≈ 111*cos(lat)
        lat_rad = lat_grid[i] * np.pi/180
        km_per_deg_lon = 111 * np.cos(lat_rad)
        km_per_deg_lat = 111
        grad_x = gx / (2*step) / km_per_deg_lon
        grad_y = gy / (2*step) / km_per_deg_lat
        grad_mag[i, j] = np.sqrt(grad_x**2 + grad_y**2)

# Assign front intensity to each original point (0-1, 0.3 °C/km = strong front)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

# Build a list of grid cells for fast nearest lookup
grid_cells = [(lat_grid[i], lon_grid[j]) for i in range(nlat) for j in range(nlon)]

# Now iterate over all surface points and store values
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

        # Find nearest grid cell for front intensity
        best_dist = np.inf
        best_intensity = 0.0
        for gi, gj in [(i, j) for i in range(nlat) for j in range(nlon)]:
            d = haversine(lat, lon, lat_grid[gi], lon_grid[gj])
            if d < best_dist:
                best_dist = d
                best_intensity = grad_mag[gi, gj] if not np.isnan(grad_mag[gi, gj]) else 0.0
        intensity = min(1.0, best_intensity / 0.3)
        front_list.append(round(intensity, 3))

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
