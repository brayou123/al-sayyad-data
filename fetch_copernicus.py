#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from copernicusmarine import subset, describe

# =====================================================
# 1. المصادقة
# =====================================================
USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

# =====================================================
# 2. تحديد الإحداثيات الصحيحة باستخدام describe
# =====================================================
print("Retrieving actual dataset coordinates...")
describe_result = describe(
    dataset_ids=["cmems_mod_med_phy-tem_anfc_4.2km_P1D-m"],
    username=USERNAME, password=PASSWORD
)
# استخراج حدود خطوط الطول والعرض من الوصف
# نأخذ القيم من أول مجموعة بيانات في النتيجة
ds_meta = describe_result[0].datasets[0]
lon_min = ds_meta.geographic_coverage.west_boundary
lon_max = ds_meta.geographic_coverage.east_boundary
lat_min = ds_meta.geographic_coverage.south_boundary
lat_max = ds_meta.geographic_coverage.north_boundary
print(f"Dataset coordinates: lat [{lat_min}, {lat_max}], lon [{lon_min}, {lon_max}]")

# =====================================================
# 3. معلمات الوقت
# =====================================================
today = datetime.utcnow().date()
yesterday = today - timedelta(days=1)
start_date = yesterday.isoformat()
end_date = yesterday.isoformat()
print(f"Fetching data for {yesterday}")

# =====================================================
# 4. دالة مساعدة لتحميل ملف وفتحه
# =====================================================
def download_and_open(dataset_id, variables, filename, depth_min=0.0, depth_max=0.0):
    print(f"Downloading {variables} from {dataset_id}...")
    path = subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min, maximum_latitude=lat_max,
        start_datetime=start_date, end_datetime=end_date,
        minimum_depth=depth_min, maximum_depth=depth_max,
        username=USERNAME, password=PASSWORD,
        output_filename=filename
    )
    # path should be a string (the filename)
    return xr.open_dataset(path)

# =====================================================
# 5. تحميل البيانات السطحية
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
# 6. تحميل ملف تعريف درجة الحرارة (أعماق متعددة)
# =====================================================
print("Downloading temperature profile (0-100m)...")
profile_path = subset(
    dataset_id="cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
    variables=["thetao"],
    minimum_longitude=lon_min, maximum_longitude=lon_max,
    minimum_latitude=lat_min, maximum_latitude=lat_max,
    start_datetime=start_date, end_datetime=end_date,
    minimum_depth=0.0, maximum_depth=100.0,
    username=USERNAME, password=PASSWORD,
    output_filename="profile.nc"
)
ds_prof = xr.open_dataset(profile_path)

# =====================================================
# 7. دمج جميع البيانات السطحية في مجموعة واحدة
# =====================================================
print("Merging surface datasets...")
# جميع البيانات السطحية تشترك في الإحداثيات (الزمن، خط العرض، خط الطول)
# نزيل أبعاد العمق (إذا كانت موجودة) ونحتفظ فقط بالمستوى السطحي.
# لكن بما أننا طلبنا depth_range=0.0..0.0، سيكون البعد عمقاً واحداً.
# نستخدم .squeeze() لإزالة البعد الزائد.
ds_surface = xr.merge([
    ds_temp.squeeze(),
    ds_sal.squeeze(),
    ds_cur.squeeze(),
    ds_chl.squeeze(),
    ds_o2.squeeze(),
    ds_kd.squeeze()
])

# =====================================================
# 8. حساب التارموكلاين لكل نقطة
# =====================================================
print("Computing thermocline...")
lons = ds_surface.longitude.values
lats = ds_surface.latitude.values
depth_vals = ds_prof.depth.values  # الأعماق المتاحة

# مصفوفة لتخزين عمق التارموكلاين (نفس أبعاد lats, lons)
thermocline_map = np.full((len(lats), len(lons)), np.nan)

for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # استخرج ملف درجة الحرارة على الأعماق لهذه النقطة
        temp_profile = ds_prof.thetao.isel(time=0, latitude=i, longitude=j).values
        # إزالة القيم المفقودة
        depths_clean = []
        temps_clean = []
        for d_idx, t in enumerate(temp_profile):
            if not np.isnan(t):
                depths_clean.append(depth_vals[d_idx])
                temps_clean.append(t)
        if len(temps_clean) < 2:
            thermocline_map[i, j] = 35.0  # قيمة افتراضية
            continue

        # حساب أقصى تدرج حراري
        max_grad = 0.0
        thermo_depth = 35.0
        for k in range(1, len(temps_clean)):
            grad = abs(temps_clean[k] - temps_clean[k-1])
            if grad > max_grad:
                max_grad = grad
                thermo_depth = (depths_clean[k] + depths_clean[k-1]) / 2
        thermocline_map[i, j] = thermo_depth

# =====================================================
# 9. بناء قائمة النقاط
# =====================================================
print("Building points list...")
points = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        # نتحقق من وجود درجة حرارة صالحة (مؤشر على بحر)
        temp_val = ds_surface.thetao.isel(time=0, latitude=i, longitude=j).values
        if np.isnan(temp_val):
            continue

        sal_val = ds_surface.so.isel(time=0, latitude=i, longitude=j).values
        u_val = ds_surface.uo.isel(time=0, latitude=i, longitude=j).values
        v_val = ds_surface.vo.isel(time=0, latitude=i, longitude=j).values
        current_speed = np.sqrt(u_val**2 + v_val**2)
        chl_val = ds_surface.chl.isel(time=0, latitude=i, longitude=j).values
        o2_val = ds_surface.o2.isel(time=0, latitude=i, longitude=j).values
        kd_val = ds_surface.kd490.isel(time=0, latitude=i, longitude=j).values

        points.append({
            "lat": float(lat),
            "lon": float(lon),
            "temperature": round(float(temp_val), 2),
            "salinity": round(float(sal_val), 2),
            "chlorophyll": round(float(chl_val) if not np.isnan(chl_val) else np.nan, 4),
            "oxygen": round(float(o2_val) if not np.isnan(o2_val) else np.nan, 2),
            "transparency": round(float(kd_val) if not np.isnan(kd_val) else np.nan, 2),
            "currentSpeed": round(float(current_speed), 3),
            "thermocline": round(float(thermocline_map[i, j]), 1)
        })

print(f"Processed {len(points)} ocean points")

# =====================================================
# 10. حفظ data.json
# =====================================================
output = {
    "timestamp": yesterday.isoformat(),
    "resolution_km": 4.2,
    "points": points
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',', ':'), ensure_ascii=False)

print("data.json saved successfully")
