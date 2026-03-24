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
# الإحداثيات الثابتة (من أخطاء سابقة)
# =====================================================
LAT_MIN = 30.1875
LAT_MAX = 45.97916793823242
LON_MIN = -6.0
LON_MAX = 36.0
DEPTH_SURFACE = 1.0182366371154785   # أقرب عمق للسطح
DEPTH_MIN_PROFILE = DEPTH_SURFACE
DEPTH_MAX_PROFILE = 100.0

# =====================================================
# الوقت
# =====================================================
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# دالة مساعدة للتحميل
# =====================================================
def download_and_open(dataset_id, variables, filename, depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE):
    print(f"Downloading {variables} from {dataset_id}...")
    path = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
        start_datetime=start_date, end_datetime=end_date,
        minimum_depth=depth_min, maximum_depth=depth_max,
        username=USERNAME, password=PASSWORD,
        output_filename=filename
    )
    # path is the file path as a string
    return xr.open_dataset(path)

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
profile_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_MIN_PROFILE, maximum_depth=DEPTH_MAX_PROFILE,
    username=USERNAME, password=PASSWORD,
    output_filename="profile.nc"
)
ds_prof = xr.open_dataset(profile_path)

# =====================================================
# دمج البيانات السطحية
# =====================================================
print("Merging surface datasets...")
# إزالة البعد العمق الزائد (الوحيد) باستخدام squeeze
ds_surface = xr.merge([
    ds_temp.squeeze(),
    ds_sal.squeeze(),
    ds_cur.squeeze(),
    ds_chl.squeeze(),
    ds_o2.squeeze(),
    ds_kd.squeeze()
])

# =====================================================
# حساب التارموكلاين
# =====================================================
print("Computing thermocline...")
lons = ds_surface.longitude.values
lats = ds_surface.latitude.values
depth_vals = ds_prof.depth.values

thermocline_map = np.full((len(lats), len(lons)), np.nan)

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # استخرج ملف درجة الحرارة على الأعماق لهذه النقطة
        temp_profile = ds_prof.thetao.isel(time=0, latitude=i, longitude=j).values
        # ازالة القيم المفقودة
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
# بناء قائمة النقاط
# =====================================================
points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = ds_surface.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_val):
            continue

        temp = float(temp_val)
        sal = float(ds_surface.so.isel(time=0, latitude=i, longitude=j).values)
        u = float(ds_surface.uo.isel(time=0, latitude=i, longitude=j).values)
        v = float(ds_surface.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u**2 + v**2)
        chl = ds_surface.chl.isel(time=0, latitude=i, longitude=j).values
        chl = float(chl) if not np.isnan(chl) else np.nan
        o2 = ds_surface.o2.isel(time=0, latitude=i, longitude=j).values
        o2 = float(o2) if not np.isnan(o2) else np.nan
        kd = ds_surface.kd490.isel(time=0, latitude=i, longitude=j).values
        kd = float(kd) if not np.isnan(kd) else np.nan

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(temp, 2),
            "salinity": round(sal, 2),
            "chlorophyll": round(chl, 4),
            "oxygen": round(o2, 2),
            "transparency": round(kd, 2),
            "currentSpeed": round(current_speed, 3),
            "thermocline": round(float(thermocline_map[i, j]), 1)
        })

print(f"Processed {len(points)} ocean points")

# =====================================================
# حفظ data.json
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
