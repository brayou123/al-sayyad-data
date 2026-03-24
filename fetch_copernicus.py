#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# =====================================================
# 1. بيانات الدخول
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# 2. معلمات الطلب
# =====================================================
LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -6.0, 36.0
DEPTH_SURFACE_MIN = 0.0
DEPTH_SURFACE_MAX = 1.0
DEPTH_PROFILE_MIN = 0.0
DEPTH_PROFILE_MAX = 100.0

today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()

print(f"Fetching data for {yesterday}")

# =====================================================
# 3. تحميل درجة الحرارة السطحية
# =====================================================
print("Downloading sea surface temperature (thetao)...")
temp_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_thetao.nc"
)
ds_thetao = xr.open_dataset(temp_path)

# =====================================================
# 4. تحميل الملوحة
# =====================================================
print("Downloading salinity (so)...")
sal_path = subset(
    dataset_id="cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
    variables=["so"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_so.nc"
)
ds_so = xr.open_dataset(sal_path)

# =====================================================
# 5. تحميل التيارات
# =====================================================
print("Downloading currents (uo, vo)...")
cur_path = subset(
    dataset_id="cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
    variables=["uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_uv.nc"
)
ds_cur = xr.open_dataset(cur_path)

# =====================================================
# 6. تحميل الكلوروفيل
# =====================================================
print("Downloading chlorophyll (chl)...")
chl_path = subset(
    dataset_id="cmems_mod_med_bgc-pft_anfc_4.2km_P1D-m",
    variables=["chl"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_chl.nc"
)
ds_chl = xr.open_dataset(chl_path)

# =====================================================
# 7. تحميل الأكسجين
# =====================================================
print("Downloading dissolved oxygen (o2)...")
o2_path = subset(
    dataset_id="cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    variables=["o2"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_o2.nc"
)
ds_o2 = xr.open_dataset(o2_path)

# =====================================================
# 8. تحميل الشفافية (kd490)
# =====================================================
print("Downloading transparency (kd490)...")
kd_path = subset(
    dataset_id="cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    variables=["kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE_MIN, maximum_depth=DEPTH_SURFACE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_kd490.nc"
)
ds_kd = xr.open_dataset(kd_path)

# =====================================================
# 9. تحميل ملف تعريف درجة الحرارة (للتارموكلاين)
# =====================================================
print("Downloading temperature profile (thetao for depths up to 100m)...")
prof_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_PROFILE_MIN, maximum_depth=DEPTH_PROFILE_MAX,
    username=USERNAME, password=PASSWORD,
    output_filename="med_thetao_profile.nc"
)
ds_prof = xr.open_dataset(prof_path)

# =====================================================
# 10. استخراج الإحداثيات (من ملف الحرارة)
# =====================================================
lons = ds_thetao.longitude.values
lats = ds_thetao.latitude.values

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # درجة حرارة السطح
        temp_surf = ds_thetao.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_surf):
            continue

        temp_val = float(temp_surf)

        # ملوحة
        sal_val = float(ds_so.so.isel(time=0, latitude=i, longitude=j).values)

        # تيارات
        u_val = float(ds_cur.uo.isel(time=0, latitude=i, longitude=j).values)
        v_val = float(ds_cur.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u_val**2 + v_val**2)

        # كلوروفيل
        chl_val = ds_chl.chl.isel(time=0, latitude=i, longitude=j).values
        chl_val = float(chl_val) if not np.isnan(chl_val) else np.nan

        # أكسجين
        o2_val = ds_o2.o2.isel(time=0, latitude=i, longitude=j).values
        o2_val = float(o2_val) if not np.isnan(o2_val) else np.nan

        # شفافية
        kd_val = ds_kd.kd490.isel(time=0, latitude=i, longitude=j).values
        kd_val = float(kd_val) if not np.isnan(kd_val) else np.nan

        # حساب التارموكلاين من ملف تعريف درجة الحرارة
        # نحصل على ملف درجة الحرارة على الأعماق لهذه النقطة
        # ds_prof له أبعاد (time, depth, latitude, longitude)
        # نستخرج العمق كمحور ونأخذ قيم الحرارة
        depth_vals = ds_prof.depth.values
        temp_profile = []
        for depth_idx, depth in enumerate(depth_vals):
            t = ds_prof.thetao.isel(time=0, depth=depth_idx, latitude=i, longitude=j).values
            if not np.isnan(t):
                temp_profile.append(float(t))
            else:
                temp_profile.append(np.nan)

        # إيجاد أقصى تدرج حراري بين الأعماق المتجاورة
        max_grad = 0.0
        thermocline_depth = 35.0  # قيمة افتراضية
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
# 11. حفظ data.json
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
