#!/usr/bin/env python3
"""
تحليل الثيرموكلين بطريقة التركيب السيني (Sigmoid Fit)
مع خطة احتياطية في حال فشل التركيب.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# بياناتك (من السطح حتى ~100 متر)
depths = np.array([1.0, 3.2, 5.5, 7.9, 10.5, 13.3, 16.3, 19.4, 22.7, 26.2, 29.9, 33.8, 37.9, 42.1, 46.7, 51.4, 56.3, 61.5, 66.9, 72.6, 78.6, 84.7, 91.2, 97.9])
temps = np.array([23.67, 22.09, 21.58, 21.27, 20.89, 20.37, 19.81, 19.30, 18.85, 18.42, 18.03, 17.67, 17.36, 17.10, 16.86, 16.67, 16.51, 16.34, 16.17, 16.02, 15.88, 15.73, 15.57, 15.41])

# دالة سينية (Sigmoid) - نفسها ولكن مع صيغة أكثر استقراراً
def sigmoid(z, T_surf, T_deep, z0, width):
    return T_deep + (T_surf - T_deep) / (1 + np.exp((z - z0) / np.abs(width) + 1e-6))

# ============================================================
# الطريقة 1: تركيب الدالة السينية (مع تحسين التخمينات)
# ============================================================

# تخمينات أولية محسّنة بناءً على بياناتك
# T_surf ~ 24 (قريب من أول نقطة)
# T_deep ~ 15 (قريب من آخر نقطة)
# z0 ~ 13 (حيث يبدأ الانخفاض الحاد)
# width ~ 5 (معامل تدريجي)
initial_guess = [24.0, 15.0, 13.0, 5.0]

# حدود دنيا وقصوى لمنع الحلول غير المنطقية
bounds = (
    [20.0, 10.0, 2.0,  0.5],   # lower bounds
    [28.0, 18.0, 30.0, 15.0]   # upper bounds
)

sigmoid_success = False
try:
    params, covariance = curve_fit(
        sigmoid,
        depths, temps,
        p0=initial_guess,
        bounds=bounds,
        method='trf',          # أكثر استقراراً من 'lm'
        maxfev=10000,          # زيادة عدد التكرارات
        ftol=1e-6,
        xtol=1e-6
    )
    sigmoid_success = True
except Exception as e:
    print(f"⚠️ فشل تركيب الدالة السينية: {e}")
    print("   سيتم استخدام الطريقة البديلة (التدرج الأقصى).")

# ============================================================
# الطريقة 2: التدرج الأقصى (بديل احتياطي)
# ============================================================

if sigmoid_success:
    T_surf, T_deep, z0, width = params
    center = z0
    # الحد العلوي = z0 - 2*width (حيث يبدأ 95% من التغير)
    top = z0 - 2 * width
    bottom = z0 + 2 * width
    thickness = 4 * width
else:
    # استخدام التدرج الأقصى لحساب المركز، ثم تقدير الحد العلوي
    grad = np.abs(np.diff(temps) / np.diff(depths))
    max_grad_idx = np.argmax(grad)
    center = (depths[max_grad_idx] + depths[max_grad_idx+1]) / 2
    # تقدير width من الفرق بين العمق الذي يسبق الانخفاض وبعده
    left_idx = max_grad_idx
    right_idx = max_grad_idx + 1
    # نبحث عن نقطة يكون فيها التغير أقل من نصف التدرج الأقصى
    half_grad = grad[max_grad_idx] * 0.5
    # تقدير تقريبي للحد العلوي (أول عمق يبدأ فيه التغير)
    top = depths[max_grad_idx] - 2.0
    # تقدير تقريبي للحد السفلي
    bottom = depths[max_grad_idx+1] + 2.0
    thickness = bottom - top
    T_surf = temps[0]
    T_deep = temps[-1]
    width = thickness / 4

# ============================================================
# المقارنة مع طريقة التدرج الأقصى (القصوى) التقليدية
# ============================================================

grad = np.abs(np.diff(temps) / np.diff(depths))
max_grad_idx = np.argmax(grad)
max_grad_depth = (depths[max_grad_idx] + depths[max_grad_idx+1]) / 2

# ============================================================
# عرض النتائج
# ============================================================

print("=" * 60)
print("نتائج تحليل الثيرموكلين")
print("=" * 60)

if sigmoid_success:
    print("\n🔵 الطريقة: تركيب الدالة السينية (Sigmoid Fit) — نجح")
else:
    print("\n🟡 الطريقة: التدرج الأقصى (بديل احتياطي)")

print("-" * 60)
print(f"  درجة حرارة السطح (T_surf)      : {T_surf:.2f} °C")
print(f"  درجة حرارة العمق (T_deep)      : {T_deep:.2f} °C")
print(f"  مركز الثيرموكلين (Center)      : {center:.1f} متر")
print(f"  ✅ الحد العلوي للطبقة (TOP)    : {top:.1f} متر  ← منطقة تجمع الأسماك")
print(f"  الحد السفلي للطبقة (Bottom)    : {bottom:.1f} متر")
print(f"  سمك الطبقة (Thickness)         : {thickness:.1f} متر")
print("-" * 60)
print("\n📊 مقارنة مع الطريقة التقليدية (التدرج الأقصى):")
print(f"  عمق التدرج الأقصى (طريقتك الحالية) : {max_grad_depth:.1f} متر")
print(f"  الفرق بين المركز والحد العلوي      : {center - top:.1f} متر")
print("=" * 60)

# ============================================================
# رسم وحفظ الصورة
# ============================================================

plt.figure(figsize=(10, 6))
plt.plot(temps, depths, 'o-', label='البيانات الأصلية', color='blue', markersize=6)

z_fit = np.linspace(0, 100, 200)
if sigmoid_success:
    T_fit = sigmoid(z_fit, *params)
    plt.plot(T_fit, z_fit, 'r-', label='منحنى التركيب (Sigmoid)', linewidth=2)

plt.axhline(y=top, color='green', linestyle=':', label=f'✅ الحد العلوي (Top) = {top:.1f} م', linewidth=2)
plt.axhline(y=center, color='orange', linestyle='--', label=f'مركز الطبقة (Center) = {center:.1f} م')
plt.axhline(y=bottom, color='purple', linestyle=':', label=f'الحد السفلي (Bottom) = {bottom:.1f} م')
plt.axhline(y=max_grad_depth, color='gray', linestyle='-.', label=f'التدرج الأقصى = {max_grad_depth:.1f} م (طريقة السكربت الحالية)')

# تظليل منطقة تجمع الأسماك
plt.axhspan(0, top, alpha=0.15, color='green', label='منطقة تجمع الأسماك (فوق الحد العلوي)')

plt.gca().invert_yaxis()
plt.xlabel('درجة الحرارة (°C)', fontsize=12)
plt.ylabel('العمق (متر)', fontsize=12)
plt.title('تحديد الثيرموكلين - مقارنة بين الطرق', fontsize=14)
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)

# حفظ الصورة
plt.savefig('thermocline_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ تم حفظ الرسم البياني كـ 'thermocline_analysis.png'")