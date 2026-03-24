#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import read_dataframe, subset

# =====================================================
# 1. تحديد المنطقة (البحر المتوسط كاملاً)
# =====================================================
LAT_MIN, LAT_MAX = 30.0, 46.0
LON_MIN, LON_MAX = -6.0, 36.0
DEPTHS = [0, 10, 20, 30, 50, 75, 100]  # لحساب الـ thermocline

# تاريخ اليوم – نأخذ أحدث تحليل متاح (أمس)
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)

# =====================================================
# 2. جلب البيانات من Copernicus (فيزياء + BGC)
# =====================================================

# --- البيانات الفيزيائية: درجة حرارة سطحية، ملوحة، تيارات سطحية ---
phy_subset = subset(
    dataset_id="cmems_mod_med_phy_anfc_4.2km_P1D-m",
    variables=["thetao", "so", "uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday.isoformat(),
    end_datetime=yesterday.isoformat(),
    minimum_depth=0, maximum_depth=0,  # سطح فقط
)
ds_phy = xr.open_dataset(phy_subset)

# --- البيانات البيوجيوكيميائية: كلوروفيل، أكسجين، شفافية ---
bgc_subset = subset(
    dataset_id="cmems_mod_med_bgc_anfc_4.2km_P1D-m",
    variables=["chl", "o2", "kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday.isoformat(),
    end_datetime=yesterday.isoformat(),
    minimum_depth=0, maximum_depth=0,
)
ds_bgc = xr.open_dataset(bgc_subset)

# --- بيانات درجة الحرارة على أعماق متعددة لحساب التارموكلاين ---
temp_profiles = subset(
    dataset_id="cmems_mod_med_phy_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=yesterday.isoformat(),
    end_datetime=yesterday.isoformat(),
    minimum_depth=min(DEPTHS), maximum_depth=max(DEPTHS),
)
ds_temp = xr.open_dataset(temp_profiles)

# =====================================================
# 3. دمج البيانات وإنشاء الشبكة
# =====================================================

# استخرج الإحداثيات
lons = ds_phy.longitude.values
lats = ds_phy.latitude.values
time = ds_phy.time.values[0]

points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # تجاهل النقاط البرية (أو استخدم mask من البيانات)
        if np.isnan(ds_phy.thetao.isel(time=0, latitude=i, longitude=j).values):
            continue

        # استخرج القيم السطحية
        temp_surf = float(ds_phy.thetao.isel(time=0, latitude=i, longitude=j).values)
        sal_surf = float(ds_phy.so.isel(time=0, latitude=i, longitude=j).values)
        u_surf = float(ds_phy.uo.isel(time=0, latitude=i, longitude=j).values)
        v_surf = float(ds_phy.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u_surf**2 + v_surf**2)

        # BGC
        chl = float(ds_bgc.chl.isel(time=0, latitude=i, longitude=j).values) if 'chl' in ds_bgc else np.nan
        o2 = float(ds_bgc.o2.isel(time=0, latitude=i, longitude=j).values) if 'o2' in ds_bgc else np.nan
        kd490 = float(ds_bgc.kd490.isel(time=0, latitude=i, longitude=j).values) if 'kd490' in ds_bgc else np.nan

        # حساب التارموكلاين (عمق أقصى تدرج حراري)
        # نجلب ملف درجة الحرارة على الأعماق المختلفة لهذه النقطة
        temp_profile = []
        for d in DEPTHS:
            # أعمق مستوى متاح
            depth_idx = np.argmin(np.abs(ds_temp.depth.values - d))
            t = float(ds_temp.thetao.isel(time=0, latitude=i, longitude=j, depth=depth_idx).values)
            temp_profile.append(t)

        # احسب التدرج بين المستويات المتجاورة
        max_grad = 0
        thermocline_depth = 35  # افتراضي
        for k in range(1, len(DEPTHS)):
            grad = abs(temp_profile[k] - temp_profile[k-1])
            if grad > max_grad:
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

# =====================================================
# 4. حفظ النتائج كـ data.json (مضغوط)
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)
