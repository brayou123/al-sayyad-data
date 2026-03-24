#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# =====================================================
# المصادقة
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# الإحداثيات الصحيحة (من رسائل التحذير)
# =====================================================
LAT_MIN = 30.1875
LAT_MAX = 45.97916793823242
LON_MIN = -5.541666507720947
LON_MAX = 36.29166793823242
DEPTH_SURFACE = 1.0182366371154785
DEPTH_MIN_PROFILE = DEPTH_SURFACE
DEPTH_MAX_PROFILE = 100.0

# =====================================================
# الوقت
# =====================================================
yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# دالة مساعدة لاستخراج المسار
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
# دالة للتحميل والفتح
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
# تحميل البيانات السطحية
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
# تحميل ملف تعريف درجة الحرارة
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
# دمج البيانات السطحية
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
# حساب التارموكلاين
# =====================================================
print("Computing thermocline...")
lats = ds_surf.latitude.values
lons = ds_surf.longitude.values
depth_vals = ds_prof.depth.values

thermocline_map = np.full((len(lats), len(lons)), np.nan)

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        point_profile = ds_prof.sel(latitude=lat, longitude=lon, method='nearest')
        temp_profile = point_profile.thetao.isel(time=0).values
        depths_clean = []
        temps_clean = []
        for d_idx, t in enumerate(temp_profile):
            if not np.isnan(t):
                depths_clean.append(depth_vals[d_idx])
                temps_clean.append(t)
        if len(temps_clean) < 2:
            thermocline_map[i, j] = 35.0
            continue
        max_grad = 0.0
        thermo_depth = 35.0
        for k in range(1, len(temps_clean)):
            grad = abs(temps_clean[k] - temps_clean[k-1])
            if grad > max_grad:
                max_grad = grad
                thermo_depth = (depths_clean[k] + depths_clean[k-1]) / 2
        thermocline_map[i, j] = thermo_depth

# =====================================================
# بناء البيانات العمودية (columnar)
# =====================================================
print("Building columnar data...")
# قوائم فارغة
lat_list = []
lon_list = []
temp_list = []
sal_list = []
chl_list = []
o2_list = []
kd_list = []
curr_list = []
thermo_list = []

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = ds_surf.thetao.isel(latitude=i, longitude=j).values
        if np.isnan(temp_val):
            continue

        # إضافة الإحداثيات
        lat_list.append(round(float(lat), 6))
        lon_list.append(round(float(lon), 6))

        # درجة الحرارة
        temp_list.append(round(float(temp_val), 2))

        # الملوحة
        sal_val = ds_surf.so.isel(latitude=i, longitude=j).values
        sal_list.append(round(float(sal_val), 2) if not np.isnan(sal_val) else None)

        # التيارات
        u_val = ds_surf.uo.isel(latitude=i, longitude=j).values
        v_val = ds_surf.vo.isel(latitude=i, longitude=j).values
        if not np.isnan(u_val) and not np.isnan(v_val):
            curr_speed = round(float(np.sqrt(u_val**2 + v_val**2)), 3)
        else:
            curr_speed = None
        curr_list.append(curr_speed)

        # كلوروفيل
        chl_val = ds_surf.chl.isel(latitude=i, longitude=j).values
        chl_list.append(round(float(chl_val), 4) if not np.isnan(chl_val) else None)

        # أكسجين
        o2_val = ds_surf.o2.isel(latitude=i, longitude=j).values
        o2_list.append(round(float(o2_val), 2) if not np.isnan(o2_val) else None)

        # شفافية
        kd_val = ds_surf.kd490.isel(latitude=i, longitude=j).values
        kd_list.append(round(float(kd_val), 2) if not np.isnan(kd_val) else None)

        # تارموكلاين
        thermo_list.append(round(float(thermocline_map[i, j]), 1))

# =====================================================
# حفظ data.json بالصيغة العمودية
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
    "thermocline": thermo_list
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print(f"Saved {len(lat_list)} points in columnar format")
print("data.json saved successfully")
