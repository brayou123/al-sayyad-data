#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# =====================================================
# 1. المصادقة
# =====================================================
USERNAME = os.environ.get("COPERNICUS_USER")
PASSWORD = os.environ.get("COPERNICUS_PASS")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials in environment")

# =====================================================
# 2. المنطقة والوقت
# =====================================================
LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -6.0, 36.0
DEPTHS = [0, 10, 20, 30, 50, 75, 100]

today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()

print(f"Fetching data for {yesterday}")

# =====================================================
# 3. البيانات الفيزيائية (حرارة، ملوحة، تيارات)
# =====================================================
phy_result = subset(
    dataset_id="MEDSEA_ANALYSISFORECAST_PHY_006_013",
    variables=["thetao", "so", "uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)

if os.path.isdir(phy_result):
    ds_phy = xr.open_dataset(phy_result, engine='zarr')
else:
    ds_phy = xr.open_dataset(phy_result, engine='netcdf4')

# =====================================================
# 4. البيانات البيوجيوكيميائية (كلوروفيل، أكسجين، شفافية)
# =====================================================
bgc_result = subset(
    dataset_id="MEDSEA_ANALYSISFORECAST_BGC_006_014",
    variables=["chl", "o2", "kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0, maximum_depth=0,
    username=USERNAME, password=PASSWORD
)

if os.path.isdir(bgc_result):
    ds_bgc = xr.open_dataset(bgc_result, engine='zarr')
else:
    ds_bgc = xr.open_dataset(bgc_result, engine='netcdf4')

# =====================================================
# 5. بيانات درجة الحرارة على أعماق متعددة (للتارموكلاين)
# =====================================================
temp_prof_result = subset(
    dataset_id="MEDSEA_ANALYSISFORECAST_PHY_006_013",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=min(DEPTHS), maximum_depth=max(DEPTHS),
    username=USERNAME, password=PASSWORD
)

if os.path.isdir(temp_prof_result):
    ds_temp = xr.open_dataset(temp_prof_result, engine='zarr')
else:
    ds_temp = xr.open_dataset(temp_prof_result, engine='netcdf4')

# =====================================================
# 6. استخراج الإحداثيات
# =====================================================
lons = ds_phy.longitude.values
lats = ds_phy.latitude.values

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_surf = ds_phy.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_surf):
            continue

        temp_surf = float(temp_surf)
        sal_surf = float(ds_phy.so.isel(time=0, latitude=i, longitude=j).values)
        u_surf = float(ds_phy.uo.isel(time=0, latitude=i, longitude=j).values)
        v_surf = float(ds_phy.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u_surf**2 + v_surf**2)

        chl = ds_bgc.chl.isel(time=0, latitude=i, longitude=j).values
        chl = float(chl) if not np.isnan(chl) else np.nan
        o2 = ds_bgc.o2.isel(time=0, latitude=i, longitude=j).values
        o2 = float(o2) if not np.isnan(o2) else np.nan
        kd490 = ds_bgc.kd490.isel(time=0, latitude=i, longitude=j).values
        kd490 = float(kd490) if not np.isnan(kd490) else np.nan

        # حساب التارموكلاين
        temp_profile = []
        for d in DEPTHS:
            depth_idx = np.argmin(np.abs(ds_temp.depth.values - d))
            t = ds_temp.thetao.isel(time=0, latitude=i, longitude=j, depth=depth_idx).values
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
            "salinity": round(sal_surf, 2),
            "chlorophyll": round(chl, 4),
            "oxygen": round(o2, 2),
            "transparency": round(kd490, 2),
            "currentSpeed": round(current_speed, 3),
            "thermocline": round(thermocline_depth, 1)
        })

print(f"Processed {len(points)} ocean points")

# =====================================================
# 7. حفظ النتائج
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
