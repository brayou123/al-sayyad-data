import numpy as np
import matplotlib
matplotlib.use('Agg')  # <- ضروري للبيئات الخادمية
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# بياناتك (من السطح حتى ~100 متر)
depths = np.array([1.0, 3.2, 5.5, 7.9, 10.5, 13.3, 16.3, 19.4, 22.7, 26.2, 29.9, 33.8, 37.9, 42.1, 46.7, 51.4, 56.3, 61.5, 66.9, 72.6, 78.6, 84.7, 91.2, 97.9])
temps = np.array([23.67, 22.09, 21.58, 21.27, 20.89, 20.37, 19.81, 19.30, 18.85, 18.42, 18.03, 17.67, 17.36, 17.10, 16.86, 16.67, 16.51, 16.34, 16.17, 16.02, 15.88, 15.73, 15.57, 15.41])

# الدالة السينية (Sigmoid)
def sigmoid(z, T_surf, T_deep, z0, width):
    return T_deep + (T_surf - T_deep) / (1 + np.exp((z - z0) / width))

# 1. تركيب الدالة
initial_guess = [24.0, 15.0, 15.0, 5.0]
params, covariance = curve_fit(sigmoid, depths, temps, p0=initial_guess, maxfev=5000)
T_surf, T_deep, z0, width = params

# 2. استخراج المعالم
center = z0
top = z0 - 2 * width
bottom = z0 + 2 * width
thickness = 4 * width

# 3. طريقة التدرج الأقصى للمقارنة
grad = np.abs(np.diff(temps) / np.diff(depths))
max_grad_idx = np.argmax(grad)
max_grad_depth = (depths[max_grad_idx] + depths[max_grad_idx+1]) / 2

# 4. عرض النتائج في السجلات (Logs)
print("=" * 50)
print("نتائج تركيب الدالة السينية (Sigmoid Fit):")
print("=" * 50)
print(f"  درجة حرارة السطح (T_surf)      : {T_surf:.2f} °C")
print(f"  درجة حرارة العمق (T_deep)      : {T_deep:.2f} °C")
print(f"  مركز الثيرموكلين (z0)          : {center:.1f} متر")
print(f"  الحد العلوي للطبقة (Top)       : {top:.1f} متر  ← منطقة تجمع الأسماك")
print(f"  الحد السفلي للطبقة (Bottom)    : {bottom:.1f} متر")
print(f"  سمك الطبقة (Thickness)         : {thickness:.1f} متر")
print("\n" + "=" * 50)
print("مقارنة مع طريقة التدرج الأقصى (القصوى):")
print("=" * 50)
print(f"  عمق التدرج الأقصى (طريقتك الحالية) : {max_grad_depth:.1f} متر")
print(f"  الفرق بين المركز والحد العلوي      : {center - top:.1f} متر")
print("=" * 50)

# 5. رسم وحفظ الصورة بدلاً من عرضها
plt.figure(figsize=(10, 6))
plt.plot(temps, depths, 'o-', label='البيانات الأصلية', color='blue')
z_fit = np.linspace(0, 100, 200)
T_fit = sigmoid(z_fit, *params)
plt.plot(T_fit, z_fit, 'r--', label='منحنى التركيب (Sigmoid)', linewidth=2)

plt.axhline(y=top, color='green', linestyle=':', label=f'الحد العلوي ({top:.1f} م)')
plt.axhline(y=center, color='orange', linestyle=':', label=f'المركز ({center:.1f} م)')
plt.axhline(y=bottom, color='purple', linestyle=':', label=f'الحد السفلي ({bottom:.1f} م)')
plt.axhline(y=max_grad_depth, color='gray', linestyle='--', label=f'التدرج الأقصى ({max_grad_depth:.1f} م)')

plt.gca().invert_yaxis()
plt.xlabel('درجة الحرارة (°C)')
plt.ylabel('العمق (متر)')
plt.title('تحديد الثيرموكلين بطريقة التركيب السيني')
plt.legend()
plt.grid(True, alpha=0.3)

# حفظ الصورة في مجلد العمل
plt.savefig('thermocline_plot.png', dpi=150)
print("\n✅ تم حفظ الرسم البياني كـ 'thermocline_plot.png'")