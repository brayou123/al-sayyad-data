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

LAT_MIN, LAT_MAX = 30.1875, 45.979
LON_MIN, LON_MAX = -6.0, 36.0
DEPTH_SURFACE = 1.02

yesterday = (datetime.utcnow().date() - timedelta(days=1)).isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# جلب درجة الحرارة (من مجموعة منفصلة)
# =====================================================
print("1. Downloading temperature...")
temp_res = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD
)
temp_path = temp_res.filenames[0] if hasattr(temp_res, 'filenames') else temp_res
ds_temp = xr.open_dataset(temp_path)

# =====================================================
# جلب الملوحة (من مجموعة منفصلة)
# =====================================================
print("2. Downloading salinity...")
sal_res = subset(
    dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    variables=["so"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD
)
sal_path = sal_res.filenames[0] if hasattr(sal_res, 'filenames') else sal_res
ds_sal = xr.open_dataset(sal_path)

# =====================================================
# جلب التيارات (من مجموعة منفصلة)
# =====================================================
print("3. Downloading currents...")
cur_res = subset(
    dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD
)
cur_path = cur_res.filenames[0] if hasattr(cur_res, 'filenames') else cur_res
ds_cur = xr.open_dataset(cur_path)

# =====================================================
# جلب الكلوروفيل والأكسجين
# =====================================================
print("4. Downloading chlorophyll and oxygen...")
bgc_res = subset(
    dataset_id="cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    variables=["chl", "o2"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD
)
bgc_path = bgc_res.filenames[0] if hasattr(bgc_res, 'filenames') else bgc_res
ds_bgc = xr.open_dataset(bgc_path)

# =====================================================
# جلب الشفافية
# =====================================================
print("5. Downloading transparency...")
opt_res = subset(
    dataset_id="cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    variables=["kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday, end_datetime=yesterday,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD
)
opt_path = opt_res.filenames[0] if hasattr(opt_res, 'filenames') else opt_res
ds_opt = xr.open_dataset(opt_path)

# =====================================================
# دمج البيانات حسب الإحداثيات
# =====================================================
lons = ds_temp.longitude.values
lats = ds_temp.latitude.values

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_val = ds_temp.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_val):
            continue

        temp = float(temp_val)
        sal = float(ds_sal.so.isel(time=0, latitude=i, longitude=j).values)
        u = float(ds_cur.uo.isel(time=0, latitude=i, longitude=j).values)
        v = float(ds_cur.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u**2 + v**2)
        chl = float(ds_bgc.chl.isel(time=0, latitude=i, longitude=j).values) if 'chl' in ds_bgc else np.nan
        o2 = float(ds_bgc.o2.isel(time=0, latitude=i, longitude=j).values) if 'o2' in ds_bgc else np.nan
        kd490 = float(ds_opt.kd490.isel(time=0, latitude=i, longitude=j).values)

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(temp, 2),
            "salinity": round(sal, 2),
            "chlorophyll": round(chl, 4),
            "oxygen": round(o2, 2),
            "transparency": round(kd490, 2),
            "currentSpeed": round(current_speed, 3),
            "thermocline": 35.0  # قيمة مؤقتة، يمكن جلبها لاحقاً
        })

print(f"Processed {len(points)} ocean points")

# =====================================================
# حفظ data.json
# =====================================================
output = {
    "timestamp": yesterday,
    "resolution_km": 4.2,
    "points": points
}
with open("data.json", "w") as f:
    json.dump(output, f)

print("data.json saved!")
