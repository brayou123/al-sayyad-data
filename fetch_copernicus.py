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
# إحداثيات البيانات الفعلية (من رسائل الخطأ)
# =====================================================
LAT_MIN = 30.1875
LAT_MAX = 45.97916793823242
LON_MIN = -6.0
LON_MAX = 36.0

# العمق السطحي المتاح هو 1.018 متر (أقرب عمق لـ 0)
DEPTH_SURFACE = 1.018
# نطاق الأعماق لملف التعريف (حتى 100 متر)
DEPTH_MIN = DEPTH_SURFACE
DEPTH_MAX = 100.0

today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()

print(f"Fetching data for {yesterday}")
print(f"Latitude bounds: {LAT_MIN} - {LAT_MAX}")
print(f"Longitude bounds: {LON_MIN} - {LON_MAX}")
print(f"Depth (surface): {DEPTH_SURFACE} m")

# =====================================================
# 1. درجة الحرارة السطحية
# =====================================================
print("Downloading sea surface temperature...")
temp_file = "med_thetao.nc"
subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=temp_file
)
ds_temp = xr.open_dataset(temp_file)

# =====================================================
# 2. الملوحة
# =====================================================
print("Downloading salinity...")
sal_file = "med_so.nc"
subset(
    dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    variables=["so"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=sal_file
)
ds_sal = xr.open_dataset(sal_file)

# =====================================================
# 3. التيارات
# =====================================================
print("Downloading currents...")
cur_file = "med_uv.nc"
subset(
    dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=cur_file
)
ds_cur = xr.open_dataset(cur_file)

# =====================================================
# 4. الكلوروفيل
# =====================================================
print("Downloading chlorophyll...")
chl_file = "med_chl.nc"
subset(
    dataset_id="cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
    variables=["chl"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=chl_file
)
ds_chl = xr.open_dataset(chl_file)

# =====================================================
# 5. الأكسجين
# =====================================================
print("Downloading dissolved oxygen...")
o2_file = "med_o2.nc"
subset(
    dataset_id="cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    variables=["o2"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=o2_file
)
ds_o2 = xr.open_dataset(o2_file)

# =====================================================
# 6. الشفافية (kd490)
# =====================================================
print("Downloading transparency...")
kd_file = "med_kd490.nc"
subset(
    dataset_id="cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    variables=["kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    output_filename=kd_file
)
ds_kd = xr.open_dataset(kd_file)

# =====================================================
# 7. ملف تعريف درجة الحرارة (للتارموكلاين)
# =====================================================
print("Downloading temperature profile...")
prof_file = "med_thetao_profile.nc"
subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_MIN, maximum_depth=DEPTH_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename=prof_file
)
ds_prof = xr.open_dataset(prof_file)

# =====================================================
# 8. استخراج الإحداثيات وإنشاء النقاط
# =====================================================
lons = ds_temp.longitude.values
lats = ds_temp.latitude.values
depth_vals = ds_prof.depth.values  # أعماق الملف التعريفي

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        temp_surf = ds_temp.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_surf):
            continue

        temp_val = float(temp_surf)
        sal_val = float(ds_sal.so.isel(time=0, latitude=i, longitude=j).values)
        u_val = float(ds_cur.uo.isel(time=0, latitude=i, longitude=j).values)
        v_val = float(ds_cur.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u_val**2 + v_val**2)

        chl_val = ds_chl.chl.isel(time=0, latitude=i, longitude=j).values
        chl_val = float(chl_val) if not np.isnan(chl_val) else np.nan
        o2_val = ds_o2.o2.isel(time=0, latitude=i, longitude=j).values
        o2_val = float(o2_val) if not np.isnan(o2_val) else np.nan
        kd_val = ds_kd.kd490.isel(time=0, latitude=i, longitude=j).values
        kd_val = float(kd_val) if not np.isnan(kd_val) else np.nan

        # حساب التارموكلاين
        temp_profile = []
        for depth_idx in range(len(depth_vals)):
            t = ds_prof.thetao.isel(time=0, depth=depth_idx, latitude=i, longitude=j).values
            if not np.isnan(t):
                temp_profile.append(float(t))
            else:
                temp_profile.append(np.nan)

        max_grad = 0.0
        thermocline_depth = 35.0
        for k in range(1, len(temp_profile)):
            if not np.isnan(temp_profile[k]) and not np.isnan(temp_profile[k-1]):
                grad = abs(temp_profile[k] - temp_profile[k-1])
                if grad > max_grad:
                    max_grad = grad
                    thermocline_depth = (depth_vals[k] + depth_vals[k-1]) / 2

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(temp_val, 2),
            "salinity": round(sal_val, 2),
            "chlorophyll": round(chl_val, 4),
            "oxygen": round(o2_val, 2),
            "transparency": round(kd_val, 2),
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
