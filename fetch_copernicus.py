#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# =====================================================
# 1. قراءة بيانات الدخول من البيئة
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials in environment")

LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -6.0, 36.0
DEPTHS = [0, 10, 20, 30, 50, 75, 100]

today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()

print(f"Fetching data for {yesterday}")

# =====================================================
# 2. جلب درجة حرارة السطح
# =====================================================
temp_surf_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)
if temp_surf_path is None:
    raise RuntimeError("subset returned None for temperature")
if os.path.isdir(temp_surf_path):
    ds_temp_surf = xr.open_dataset(temp_surf_path, engine='zarr')
else:
    ds_temp_surf = xr.open_dataset(temp_surf_path, engine='netcdf4')

# =====================================================
# 3. جلب الملوحة
# =====================================================
salinity_path = subset(
    dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    variables=["so"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)
if salinity_path is None:
    raise RuntimeError("subset returned None for salinity")
if os.path.isdir(salinity_path):
    ds_sal = xr.open_dataset(salinity_path, engine='zarr')
else:
    ds_sal = xr.open_dataset(salinity_path, engine='netcdf4')

# =====================================================
# 4. جلب التيارات
# =====================================================
current_path = subset(
    dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)
if current_path is None:
    raise RuntimeError("subset returned None for currents")
if os.path.isdir(current_path):
    ds_cur = xr.open_dataset(current_path, engine='zarr')
else:
    ds_cur = xr.open_dataset(current_path, engine='netcdf4')

# =====================================================
# 5. جلب الكلوروفيل والأكسجين
# =====================================================
bgc_bio_path = subset(
    dataset_id="cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    variables=["chl", "o2"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)
if bgc_bio_path is None:
    raise RuntimeError("subset returned None for bgc-bio")
if os.path.isdir(bgc_bio_path):
    ds_bio = xr.open_dataset(bgc_bio_path, engine='zarr')
else:
    ds_bio = xr.open_dataset(bgc_bio_path, engine='netcdf4')

# =====================================================
# 6. جلب الشفافية (kd490)
# =====================================================
optics_path = subset(
    dataset_id="cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    variables=["kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)
if optics_path is None:
    raise RuntimeError("subset returned None for optics")
if os.path.isdir(optics_path):
    ds_opt = xr.open_dataset(optics_path, engine='zarr')
else:
    ds_opt = xr.open_dataset(optics_path, engine='netcdf4')

# =====================================================
# 7. جلب درجة الحرارة على الأعماق
# =====================================================
temp_profile_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=min(DEPTHS), maximum_depth=max(DEPTHS),
    username=USERNAME, password=PASSWORD
)
if temp_profile_path is None:
    raise RuntimeError("subset returned None for temperature profile")
if os.path.isdir(temp_profile_path):
    ds_temp_prof = xr.open_dataset(temp_profile_path, engine='zarr')
else:
    ds_temp_prof = xr.open_dataset(temp_profile_path, engine='netcdf4')

# =====================================================
# 8. استخراج الإحداثيات وإنشاء النقاط
# =====================================================
lons = ds_temp_surf.longitude.values
lats = ds_temp_surf.latitude.values

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_surf = ds_temp_surf.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_surf):
            continue

        temp_surf = float(temp_surf)
        sal = float(ds_sal.so.isel(time=0, latitude=i, longitude=j).values)
        u = float(ds_cur.uo.isel(time=0, latitude=i, longitude=j).values)
        v = float(ds_cur.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u**2 + v**2)

        chl = ds_bio.chl.isel(time=0, latitude=i, longitude=j).values
        chl = float(chl) if not np.isnan(chl) else np.nan
        o2 = ds_bio.o2.isel(time=0, latitude=i, longitude=j).values
        o2 = float(o2) if not np.isnan(o2) else np.nan
        kd490 = ds_opt.kd490.isel(time=0, latitude=i, longitude=j).values
        kd490 = float(kd490) if not np.isnan(kd490) else np.nan

        temp_profile = []
        for d in DEPTHS:
            depth_idx = np.argmin(np.abs(ds_temp_prof.depth.values - d))
            t = ds_temp_prof.thetao.isel(time=0, latitude=i, longitude=j, depth=depth_idx).values
            t = float(t) if not np.isnan(t) else np.nan
            temp_profile.append(t)

        max_grad = 0
        thermocline_depth = 35.0
        for k in range(1, len(DEPTHS)):
            grad = abs(temp_profile[k] - temp_profile[k-1])
            if grad > max_grad and not (np.isnan(temp_profile[k]) or np.isnan(temp_profile[k-1])):
                max_grad = grad
                thermocline_depth = (DEPTHS[k] + DEPTHS[k-1]) / 2

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(temp_surf, 2),
            "salinity": round(sal, 2),
            "chlorophyll": round(chl, 4),
            "oxygen": round(o2, 2),
            "transparency": round(kd490, 2),
            "currentSpeed": round(current_speed, 3),
            "thermocline": round(thermocline_depth, 1)
        })

print(f"Processed {len(points)} ocean points")

# =====================================================
# 9. حفظ data.json
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
