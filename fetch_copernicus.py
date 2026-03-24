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
# الإحداثيات
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
# حساب التارموكلاين — بدون حلقة (numpy vectorized)
# =====================================================
print("Computing thermocline (vectorized)...")
depth_vals = ds_prof.depth.values

# شكل البيانات: (time, depth, lat, lon)
temp_4d = ds_prof.thetao.values  # (time, depth, lat, lon)
temp_3d = temp_4d[0]             # (depth, lat, lon) — أول وقت فقط

# حساب التدرج على محور العمق
grad = np.abs(np.diff(temp_3d, axis=0))  # (depth-1, lat, lon)

# إيجاد أعمق تدرج أقصى
thermo_idx = np.argmax(grad, axis=0)     # (lat, lon)

# حساب عمق التارموكلاين كمتوسط بين طبقتين
depth_upper = depth_vals[:-1]
depth_lower = depth_vals[1:]
thermocline_map = (depth_upper[thermo_idx] + depth_lower[thermo_idx]) / 2

# نقاط بدون بيانات → NaN
all_nan_mask = np.all(np.isnan(temp_3d), axis=0)
thermocline_map = np.where(all_nan_mask, np.nan, thermocline_map)

# =====================================================
# بناء البيانات العمودية (columnar)
# =====================================================
print("Building columnar data...")
lats = ds_surf.latitude.values
lons = ds_surf.longitude.values

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
        temp_val = float(ds_surf.thetao.isel(latitude=i, longitude=j).values)
        if np.isnan(temp_val):
            continue

        lat_list.append(round(float(lat), 6))
        lon_list.append(round(float(lon), 6))

        # درجة الحرارة (°C) — بدون تغيير
        temp_list.append(round(temp_val, 2))

        # الملوحة (ppt) — بدون تغيير
        sal_val = float(ds_surf.so.isel(latitude=i, longitude=j).values)
        sal_list.append(round(sal_val, 2) if not np.isnan(sal_val) else None)

        # التيار: m/s → knots (× 1.944)
        u_val = float(ds_surf.uo.isel(latitude=i, longitude=j).values)
        v_val = float(ds_surf.vo.isel(latitude=i, longitude=j).values)
        if not np.isnan(u_val) and not np.isnan(v_val):
            curr_knots = round(np.sqrt(u_val**2 + v_val**2) * 1.944, 2)
        else:
            curr_knots = None
        curr_list.append(curr_knots)

        # كلوروفيل (mg/m³) — بدون تغيير
        chl_val = float(ds_surf.chl.isel(latitude=i, longitude=j).values)
        chl_list.append(round(chl_val, 4) if not np.isnan(chl_val) else None)

        # الأكسجين: mmol/m³ → ml/l (÷ 44.661)
        o2_val = float(ds_surf.o2.isel(latitude=i, longitude=j).values)
        if not np.isnan(o2_val):
            o2_ml = round(o2_val / 44.661, 2)
        else:
            o2_ml = None
        o2_list.append(o2_ml)

        # الشفافية: kd490 m⁻¹ → Secchi depth m (1.7 ÷ kd490)
        kd_val = float(ds_surf.kd490.isel(latitude=i, longitude=j).values)
        if not np.isnan(kd_val) and kd_val > 0.01:
            secchi = round(1.7 / kd_val, 1)
        else:
            secchi = None
        kd_list.append(secchi)

        # التارموكلاين (m) — من الخريطة المحسوبة مسبقاً
        thermo_val = thermocline_map[i, j]
        thermo_list.append(round(float(thermo_val), 1) if not np.isnan(thermo_val) else None)

# =====================================================
# حفظ data.json
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

print(f"Saved {len(lat_list)} points")
print("data.json saved successfully")
print(f"Sample oxygen (ml/l): {o2_list[:3]}")
print(f"Sample transparency/Secchi (m): {kd_list[:3]}")
print(f"Sample currentSpeed (knots): {curr_list[:3]}")
