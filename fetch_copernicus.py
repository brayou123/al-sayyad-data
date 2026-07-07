#!/usr/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from scipy.interpolate import RegularGridInterpolator
from copernicusmarine import subset

USERNAME = os.environ.get("COPERNICUSMARINE_USERNAME")
PASSWORD = os.environ.get("COPERNICUSMARINE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise ValueError("Missing Copernicus credentials")

LAT_MIN = 36.5
LAT_MAX = 37.0
LON_MIN = -1.7
LON_MAX =  8.6
DEPTH_SURFACE = 1.0182366371154785
DEPTH_MAX     = 150.0
GRID_STEP     = 0.04

# استخدام تاريخ قبل يومين لضمان توفر البيانات في جميع المجموعات (NRT و MY)
target_date = (datetime.utcnow().date() - timedelta(days=2))
yesterday = target_date.isoformat()
print(f"Date : {yesterday}")

SPECIES = [
    {
        "id": "sardine", "name": "Sardine",
        "sci": "Sardina pilchardus", "color": "#00aaff",
        "depth_min": 20, "depth_max": 300,
        "ranges": {
            "temperature":  (13.0, 22.0),
            "chlorophyll":  (0.05, 3.0),
            "oxygen":       (5.0,  8.5),
            "salinity":     (36.0, 38.5),
            "transparency": (5.0,  35.0),
            "currentSpeed": (0.0,  1.5),
            "thermocline":  (15.0, 50.0),
        },
        "weights": {
            "temperature":  0.30,
            "chlorophyll":  0.25,
            "oxygen":       0.15,
            "salinity":     0.10,
            "transparency": 0.10,
            "currentSpeed": 0.05,
            "thermocline":  0.05,
        },
        "critical": ["temperature", "oxygen"],
    },
    {
        "id": "anchovy", "name": "Anchovy",
        "sci": "Engraulis encrasicolus", "color": "#ff4455",
        "depth_min": 15, "depth_max": 250,
        "ranges": {
            "temperature":  (14.0, 23.0),
            "chlorophyll":  (0.1,  5.0),
            "oxygen":       (5.0,  8.0),
            "salinity":     (36.0, 38.5),
            "transparency": (5.0,  25.0),
            "currentSpeed": (0.0,  1.5),
            "thermocline":  (15.0, 45.0),
        },
        "weights": {
            "temperature":  0.30,
            "chlorophyll":  0.25,
            "oxygen":       0.15,
            "salinity":     0.10,
            "transparency": 0.10,
            "currentSpeed": 0.05,
            "thermocline":  0.05,
        },
        "critical": ["temperature", "oxygen"],
    },
    {
        "id": "lacha", "name": "Lacha",
        "sci": "Sardinella aurita", "color": "#22cc77",
        "depth_min": 15, "depth_max": 250,
        "ranges": {
            "temperature":  (20.0, 28.0),
            "chlorophyll":  (0.05, 2.5),
            "oxygen":       (4.5,  8.0),
            "salinity":     (36.0, 39.0),
            "transparency": (5.0,  30.0),
            "currentSpeed": (0.0,  2.0),
            "thermocline":  (20.0, 55.0),
        },
        "weights": {
            "temperature":  0.35,
            "chlorophyll":  0.20,
            "oxygen":       0.15,
            "salinity":     0.10,
            "transparency": 0.10,
            "currentSpeed": 0.05,
            "thermocline":  0.05,
        },
        "critical": ["temperature", "oxygen"],
    },
]

HSI_THRESHOLD = 0.60

def get_path(result):
    if isinstance(result, str): return result
    if hasattr(result, 'filenames') and result.filenames: return result.filenames[0]
    if hasattr(result, 'filename'): return result.filename
    raise TypeError(f"Cannot extract path from {type(result)}")

def download(dataset_id, variables, filename,
             depth_min=DEPTH_SURFACE, depth_max=DEPTH_SURFACE):
    print(f"  Downloading {variables} ...")
    result = subset(
        dataset_id=dataset_id, variables=variables,
        minimum_longitude=LON_MIN, maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,  maximum_latitude=LAT_MAX,
        start_datetime=yesterday,  end_datetime=yesterday,
        minimum_depth=depth_min,   maximum_depth=depth_max,
        username=USERNAME,         password=PASSWORD,
        output_filename=filename
    )
    path = get_path(result)
    size = os.path.getsize(path)
    if size == 0: raise RuntimeError(f"{filename} is empty")
    print(f"    OK {size/1024:.1f} KB")
    return xr.open_dataset(path, engine='netcdf4')

def get_latlon(ds):
    lat = next((n for n in ['latitude','lat'] if n in ds.dims), None)
    lon = next((n for n in ['longitude','lon'] if n in ds.dims), None)
    return lat, lon

def regrid_2d(data_2d, src_lat, src_lon, tgt_lat, tgt_lon):
    interp = RegularGridInterpolator(
        (src_lat, src_lon), data_2d,
        method='linear', bounds_error=False, fill_value=np.nan
    )
    la, lo = np.meshgrid(tgt_lat, tgt_lon, indexing='ij')
    return interp(np.stack([la.ravel(), lo.ravel()], axis=-1)).reshape(len(tgt_lat), len(tgt_lon))

def extract_surface(ds, var):
    data = ds[var].values
    while data.ndim > 2:
        data = data[0]
    return data

def extract_profile(ds, var):
    data = ds[var].values
    if data.ndim == 4:
        data = data[0]
    return data

def get_profile_val(profile_3d, depth_arr, src_lat, src_lon, target_depth, i, j):
    si = np.argmin(np.abs(src_lat - tgt_lat[i]))
    sj = np.argmin(np.abs(src_lon - tgt_lon[j]))
    idx_d = np.argmin(np.abs(depth_arr - target_depth))
    val = profile_3d[idx_d, si, sj]
    return float(val) if not np.isnan(val) else None

print("\n=== Downloading ===")

ds_sst  = download("SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_c_V2",
                   ["analysed_sst"], "sst.nc")

# تم التعديل هنا: استخدام الإصدار NRT بدلاً من MY لضمان أحدث البيانات
ds_chl  = download("cmems_obs-oc_med_bgc-plankton_nrt_l4-gapfree-multi-1km_P1D",
                   ["CHL"], "chl.nc")

ds_o2   = download("cmems_mod_med_bgc-bio_anfc_4.2km_P1D-m",
                   ["o2"], "o2.nc",
                   depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX)

ds_kd   = download("cmems_mod_med_bgc-optics_anfc_4.2km_P1D-m",
                   ["kd490"], "kd.nc")

ds_cur  = download("cmems_mod_med_phy-cur_anfc_4.2km_P1D-m",
                   ["uo","vo"], "cur.nc",
                   depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX)

ds_sal  = download("cmems_mod_med_phy-sal_anfc_4.2km_P1D-m",
                   ["so"], "sal.nc",
                   depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX)

ds_tem  = download("cmems_mod_med_phy-tem_anfc_4.2km_P1D-m",
                   ["thetao"], "tem.nc",
                   depth_min=DEPTH_SURFACE, depth_max=DEPTH_MAX)

tgt_lat = np.arange(LAT_MIN, LAT_MAX + GRID_STEP, GRID_STEP)
tgt_lon = np.arange(LON_MIN, LON_MAX + GRID_STEP, GRID_STEP)
NL, NO = len(tgt_lat), len(tgt_lon)
print(f"\nGrid: {NL} x {NO} = {NL*NO:,} points")

print("\n=== Processing ===")

lat_n, lon_n = get_latlon(ds_sst)
sst_2d = regrid_2d(extract_surface(ds_sst, "analysed_sst"),
                   ds_sst[lat_n].values, ds_sst[lon_n].values, tgt_lat, tgt_lon)
if np.nanmean(sst_2d) > 100:
    sst_2d -= 273.15

lat_n, lon_n = get_latlon(ds_chl)
chl_2d = regrid_2d(extract_surface(ds_chl, "CHL"),
                   ds_chl[lat_n].values, ds_chl[lon_n].values, tgt_lat, tgt_lon)

lat_n, lon_n = get_latlon(ds_kd)
kd_2d  = regrid_2d(extract_surface(ds_kd, "kd490"),
                   ds_kd[lat_n].values, ds_kd[lon_n].values, tgt_lat, tgt_lon)
sec_2d = np.where((kd_2d > 0.01) & ~np.isnan(kd_2d), 1.7 / kd_2d, np.nan)

lat_n, lon_n = get_latlon(ds_o2)
src_lat_o2  = ds_o2[lat_n].values
src_lon_o2  = ds_o2[lon_n].values
dep_o2      = ds_o2['depth'].values
prof_o2     = extract_profile(ds_o2, "o2")

lat_n, lon_n = get_latlon(ds_cur)
src_lat_cur = ds_cur[lat_n].values
src_lon_cur = ds_cur[lon_n].values
dep_cur     = ds_cur['depth'].values
prof_uo     = extract_profile(ds_cur, "uo")
prof_vo     = extract_profile(ds_cur, "vo")

lat_n, lon_n = get_latlon(ds_sal)
src_lat_sal = ds_sal[lat_n].values
src_lon_sal = ds_sal[lon_n].values
dep_sal     = ds_sal['depth'].values
prof_sal    = extract_profile(ds_sal, "so")

lat_n, lon_n = get_latlon(ds_tem)
src_lat_tem = ds_tem[lat_n].values
src_lon_tem = ds_tem[lon_n].values
dep_tem     = ds_tem['depth'].values
prof_tem    = extract_profile(ds_tem, "thetao")

grad_tem    = np.abs(np.diff(prof_tem, axis=0))
thermo_idx  = np.argmax(grad_tem, axis=0)
d_up        = dep_tem[:-1]
d_dn        = dep_tem[1:]
thermo_raw  = (d_up[thermo_idx] + d_dn[thermo_idx]) / 2.0
all_nan_tem = np.all(np.isnan(prof_tem), axis=0)
thermo_raw  = np.where(all_nan_tem, np.nan, thermo_raw)
thermo_2d   = regrid_2d(thermo_raw, src_lat_tem, src_lon_tem, tgt_lat, tgt_lon)

valid_tem  = ~np.isnan(prof_tem)
n_valid    = np.sum(valid_tem, axis=0)
max_d_idx  = np.clip(n_valid - 1, 0, len(dep_tem)-1)
bathy_raw  = dep_tem[max_d_idx]
bathy_raw  = np.where(n_valid == 0, np.nan, bathy_raw)
bathy_2d   = regrid_2d(bathy_raw, src_lat_tem, src_lon_tem, tgt_lat, tgt_lon)

print("Processing done.")

def compute_si(val, lo, hi):
    if val is None or np.isnan(float(val)): return None
    val = float(val)
    rang  = hi - lo or 1.0
    decay = rang * 0.30
    if val < lo: return max(0.0, 1.0 - (lo - val) / decay)
    if val > hi: return max(0.0, 1.0 - (val - hi) / decay)
    center = (lo + hi) / 2.0
    return 0.70 + 0.30 * (1.0 - abs(val - center) / (rang / 2.0))

def compute_hsi(point, sp, bathy):
    ranges   = sp["ranges"]
    weights  = sp["weights"]
    critical = sp["critical"]
    if bathy is not None and not np.isnan(float(bathy)):
        if float(bathy) < sp["depth_min"] or float(bathy) > sp["depth_max"]:
            return 0.0, []
    for key in critical:
        v = point.get(key)
        if v is None or np.isnan(float(v)): return 0.0, []
    si_vals = {}
    for key, (lo, hi) in ranges.items():
        si = compute_si(point.get(key), lo, hi)
        si_vals[key] = si if si is not None else 0.35
    total_w = sum(weights.values())
    hsi = sum(weights[k] * si_vals[k] for k in weights) / total_w
    labels = {
        "temperature":"Temp","chlorophyll":"Chl","oxygen":"O₂",
        "salinity":"Sal","transparency":"Secchi",
        "currentSpeed":"Current","thermocline":"Thermocline",
    }
    matched = [labels[k] for k, v in si_vals.items() if v >= 0.80]
    return round(float(hsi), 4), matched

print("\n=== Computing HSI ===")
hotspots  = []
total_pts = 0
passed_pts = 0

for i in range(NL):
    for j in range(NO):
        t_surf = sst_2d[i, j]
        if np.isnan(t_surf): continue
        total_pts += 1
        thermo    = thermo_2d[i, j]
        bathy     = bathy_2d[i, j]
        ref_depth = float(thermo) if not np.isnan(thermo) else 30.0
        ref_depth = min(ref_depth, DEPTH_MAX)
        o2_val  = get_profile_val(prof_o2,  dep_o2,  src_lat_o2,  src_lon_o2,  ref_depth, i, j)
        uo_val  = get_profile_val(prof_uo,  dep_cur, src_lat_cur, src_lon_cur, ref_depth, i, j)
        vo_val  = get_profile_val(prof_vo,  dep_cur, src_lat_cur, src_lon_cur, ref_depth, i, j)
        sal_val = get_profile_val(prof_sal, dep_sal, src_lat_sal, src_lon_sal, ref_depth, i, j)
        tem_val = get_profile_val(prof_tem, dep_tem, src_lat_tem, src_lon_tem, ref_depth, i, j)
        if o2_val is not None: o2_val = o2_val / 44.661
        cur_val = None
        if uo_val is not None and vo_val is not None:
            cur_val = float(np.sqrt(uo_val**2 + vo_val**2)) * 1.944
        point = {
            "temperature":  float(tem_val) if tem_val is not None else float(t_surf),
            "chlorophyll":  float(chl_2d[i,j]) if not np.isnan(chl_2d[i,j]) else None,
            "oxygen":       o2_val,
            "salinity":     sal_val,
            "transparency": float(sec_2d[i,j]) if not np.isnan(sec_2d[i,j]) else None,
            "currentSpeed": cur_val,
            "thermocline":  float(thermo)       if not np.isnan(thermo)       else None,
        }
        b = float(bathy) if not np.isnan(bathy) else None
        best_score = 0.0
        best_sp    = None
        all_sp     = []
        for sp in SPECIES:
            score, matched = compute_hsi(point, sp, b)
            if score >= HSI_THRESHOLD:
                all_sp.append({
                    "id":    sp["id"],
                    "name":  sp["name"],
                    "color": sp["color"],
                    "score": round(score * 100, 1),
                    "vars":  matched,
                })
                if score > best_score:
                    best_score = score
                    best_sp    = sp
        if not all_sp: continue
        passed_pts += 1
        all_sp.sort(key=lambda x: x["score"], reverse=True)
        hotspots.append({
            "lat":     round(float(tgt_lat[i]), 5),
            "lon":     round(float(tgt_lon[j]), 5),
            "score":   round(best_score * 100, 1),
            "color":   best_sp["color"],
            "name":    best_sp["name"],
            "depth":   round(ref_depth, 1),
            "species": all_sp,
            "data": {
                "temp":    round(point["temperature"], 2),
                "chl":     round(point["chlorophyll"], 3)  if point["chlorophyll"]  else None,
                "o2":      round(point["oxygen"], 2)       if point["oxygen"]       else None,
                "sal":     round(point["salinity"], 2)     if point["salinity"]     else None,
                "secchi":  round(point["transparency"], 1) if point["transparency"] else None,
                "current": round(point["currentSpeed"], 2) if point["currentSpeed"] else None,
                "thermo":  round(point["thermocline"], 1)  if point["thermocline"]  else None,
                "bathy":   round(b, 0)                     if b                     else None,
            }
        })

hotspots.sort(key=lambda x: x["score"], reverse=True)
print(f"Total sea points : {total_pts:,}")
print(f"Hotspots (>=60%) : {passed_pts:,}")

output = {"timestamp": yesterday, "count": len(hotspots), "hotspots": hotspots}
with open("hotspots.json", "w", encoding="utf-8") as f:
    json.dump(output, f, separators=(',',':'), ensure_ascii=False)

size_kb = os.path.getsize("hotspots.json") / 1024
print(f"Saved hotspots.json — {len(hotspots):,} hotspots — {size_kb:.1f} KB")
print("Done.")