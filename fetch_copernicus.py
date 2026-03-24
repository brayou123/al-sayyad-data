#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset

# ------------------------------
# 1. التحقق من بيانات الدخول
# ------------------------------
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials in environment")

# ------------------------------
# 2. الإحداثيات والوقت
# ------------------------------
LAT_MIN, LAT_MAX = 30.1875, 45.979      # حدود خطوط العرض الحقيقية
LON_MIN, LON_MAX = -6.0, 36.0           # الحدود الغربية والشرقية
DEPTH_SURFACE = 1.02                    # أقرب عمق للسطح (متوفر في البيانات)

today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()

print(f"Fetching data for {yesterday}")

# ------------------------------
# 3. دالة مساعدة لتحميل الملف
# ------------------------------
def get_file_path(result):
    """استخراج مسار الملف من نتيجة subset (حسب الإصدار)."""
    if hasattr(result, 'filenames') and result.filenames:
        return result.filenames[0]
    if isinstance(result, str):
        return result
    raise TypeError(f"Cannot extract path from {type(result)}")

# ------------------------------
# 4. تحميل البيانات الفيزيائية (حرارة، ملوحة، تيارات)
# ------------------------------
print("Downloading physical data (temperature, salinity, currents)...")
phy_res = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao", "so", "uo", "vo"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    dry_run=False
)
phy_path = get_file_path(phy_res)
ds_phy = xr.open_dataset(phy_path)

# ------------------------------
# 5. تحميل البيانات البيوجيوكيميائية (كلوروفيل، أكسجين)
# ------------------------------
print("Downloading biogeochemical data (chlorophyll, oxygen)...")
bgc_res = subset(
    dataset_id="cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
    variables=["chl", "o2"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    dry_run=False
)
bgc_path = get_file_path(bgc_res)
ds_bgc = xr.open_dataset(bgc_path)

# ------------------------------
# 6. تحميل بيانات الشفافية (kd490)
# ------------------------------
print("Downloading optics data (transparency kd490)...")
opt_res = subset(
    dataset_id="cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
    variables=["kd490"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=DEPTH_SURFACE, maximum_depth=DEPTH_SURFACE,
    username=USERNAME, password=PASSWORD,
    dry_run=False
)
opt_path = get_file_path(opt_res)
ds_opt = xr.open_dataset(opt_path)

# ------------------------------
# 7. تحميل بيانات درجة الحرارة على أعماق متعددة (للتارموكلاين)
# ------------------------------
print("Downloading temperature profile (for thermocline)...")
prof_res = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
    minimum_latitude=LAT_MIN, maximum_latitude=LAT_MAX,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=1.02, maximum_depth=100.0,   # نطاق الأعماق
    username=USERNAME, password=PASSWORD,
    dry_run=False
)
prof_path = get_file_path(prof_res)
ds_prof = xr.open_dataset(prof_path)

# ------------------------------
# 8. استخراج الإحداثيات
# ------------------------------
lons = ds_phy.longitude.values
lats = ds_phy.latitude.values

# ------------------------------
# 9. حساب التارموكلاين لكل نقطة
# ------------------------------
def compute_thermocline(temp_vals, depths):
    """حساب عمق التارموكلاين (أقصى تدرج حراري)."""
    if len(temp_vals) < 2:
        return 35.0
    max_grad = 0.0
    thermocline_depth = 35.0
    for i in range(1, len(temp_vals)):
        grad = abs(temp_vals[i] - temp_vals[i-1])
        if grad > max_grad:
            max_grad = grad
            thermocline_depth = (depths[i] + depths[i-1]) / 2
    return thermocline_depth

# ------------------------------
# 10. تجميع النقاط
# ------------------------------
points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # درجة حرارة السطح (مؤشر على وجود بحر)
        temp_surf = ds_phy.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_surf):
            continue

        # قراءة القيم السطحية
        temp_surf_val = float(temp_surf)
        salinity = float(ds_phy.so.isel(time=0, latitude=i, longitude=j).values)
        u = float(ds_phy.uo.isel(time=0, latitude=i, longitude=j).values)
        v = float(ds_phy.vo.isel(time=0, latitude=i, longitude=j).values)
        current_speed = np.sqrt(u*u + v*v)

        # BGC
        chl = ds_bgc.chl.isel(time=0, latitude=i, longitude=j).values
        chl = float(chl) if not np.isnan(chl) else np.nan
        o2 = ds_bgc.o2.isel(time=0, latitude=i, longitude=j).values
        o2 = float(o2) if not np.isnan(o2) else np.nan
        kd490 = ds_opt.kd490.isel(time=0, latitude=i, longitude=j).values
        kd490 = float(kd490) if not np.isnan(kd490) else np.nan

        # حساب التارموكلاين من ملف التعريف
        # استخرج ملف درجة الحرارة على الأعماق لهذه النقطة
        depth_vals = ds_prof.depth.values
        temp_profile = []
        for d in depth_vals:
            t = ds_prof.thetao.isel(time=0, latitude=i, longitude=j, depth=...).values
            # هنا نحتاج إلى فهرس العمق الصحيح – الأسهل استخدام .sel
            # لكن للتبسيط سنأخذ جميع الأعماق
        # الطريقة الأسهل: استخدام .sel مع depth متغير
        # لكن في xarray .sel يتطلب إحداثيات محددة، لذا نستخدم isel مع البحث عن أقرب عمق
        depth_idx = np.arange(len(depth_vals))
        temp_profile = []
        for idx in depth_idx:
            t = ds_prof.thetao.isel(time=0, latitude=i, longitude=j, depth=idx).values
            if not np.isnan(t):
                temp_profile.append(float(t))
            else:
                temp_profile.append(np.nan)

        # احسب التارموكلاين من الملف الحراري
        thermocline = compute_thermocline(temp_profile, depth_vals)

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(temp_surf_val, 2),
            "salinity": round(salinity, 2),
            "chlorophyll": round(chl, 4),
            "oxygen": round(o2, 2),
            "transparency": round(kd490, 2),
            "currentSpeed": round(current_speed, 3),
            "thermocline": round(thermocline, 1)
        })

print(f"Processed {len(points)} ocean points")

# ------------------------------
# 11. حفظ النتيجة
# ------------------------------
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
