"""
YZM0206 - Lab05 Uygulama 1: Noktasal İşlemler (Point Operations) ve Renk Uzayları
==================================================================================
Bu modülde lena.jpg görüntüsü üzerinde:
  - RGB, BGR, YCrCb ve HSV renk uzayı dönüşümleri
  - Negatif alma: g(x,y) = 255 - f(x,y)
  - Histogram çizimi
  - İstatistiksel analiz (Min, Max, Median, Mean)
işlemleri gerçekleştirilmektedir.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 0. Çıktı dizinini oluştur
# ============================================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Görüntüyü Oku
# ============================================================
# OpenCV görüntüyü varsayılan olarak BGR formatında okur
img_bgr = cv.imread("lena.jpg")

if img_bgr is None:
    raise FileNotFoundError("lena.jpg bulunamadı! Dosyanın proje dizininde olduğundan emin olun.")

print(f"Görüntü boyutu  : {img_bgr.shape}")          # (512, 512, 3)
print(f"Veri tipi        : {img_bgr.dtype}")           # uint8
print(f"Piksel değer aralığı: [{img_bgr.min()}, {img_bgr.max()}]")

# ============================================================
# 2. Renk Uzayı Dönüşümleri
# ============================================================
# BGR -> RGB dönüşümü (matplotlib RGB bekler)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# BGR -> YCrCb dönüşümü
# Y: Parlaklık (luminance), Cr: Kırmızı-fark, Cb: Mavi-fark
img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)

# BGR -> HSV dönüşümü
# H: Renk tonu (Hue), S: Doygunluk (Saturation), V: Değer (Value/Brightness)
img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

# --- Görselleştirme: 4 renk uzayı yan yana ---
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Renk Uzayı Dönüşümleri", fontsize=16, fontweight="bold")

# RGB
axes[0].imshow(img_rgb)
axes[0].set_title("RGB")
axes[0].axis("off")

# BGR (matplotlib RGB beklediğinden renkler yer değiştirir -> farkı görmek için)
axes[1].imshow(img_bgr)  # matplotlib bunu RGB olarak yorumlar, B↔R yer değiştirir
axes[1].set_title("BGR (matplotlib'de yanlış renk)")
axes[1].axis("off")

# YCrCb
axes[2].imshow(img_ycrcb)
axes[2].set_title("YCrCb")
axes[2].axis("off")

# HSV
axes[3].imshow(img_hsv)
axes[3].set_title("HSV")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "renk_uzaylari.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 3. Negatif Alma: g(x,y) = 255 - f(x,y)
# ============================================================
# Negatif işlemi her piksel değerini 255'ten çıkarır.
# Bu, lineer cebir açısından bir afin dönüşümdür: g = -f + 255·1
# Matris formunda: G = 255·J - F  (J: tüm elemanları 1 olan matris)
img_negative = 255 - img_rgb  # NumPy broadcasting ile vektörel işlem

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Negatif Alma: g(x,y) = 255 - f(x,y)", fontsize=16, fontweight="bold")

axes[0].imshow(img_rgb)
axes[0].set_title("Orijinal (RGB)")
axes[0].axis("off")

axes[1].imshow(img_negative)
axes[1].set_title("Negatif")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "negatif.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 4. Histogram Çizimi
# ============================================================
# Grayscale histogram
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Histogram Analizi", fontsize=16, fontweight="bold")

# Grayscale histogram
axes[0].hist(img_gray.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.8)
axes[0].set_title("Grayscale Histogram")
axes[0].set_xlabel("Piksel Değeri (0-255)")
axes[0].set_ylabel("Frekans")
axes[0].grid(True, alpha=0.3)

# RGB kanal histogramları
colors = ("red", "green", "blue")
channel_names = ("R", "G", "B")
for i, (color, name) in enumerate(zip(colors, channel_names)):
    hist = cv.calcHist([img_rgb], [i], None, [256], [0, 256])
    axes[1].plot(hist, color=color, label=f"{name} Kanalı", alpha=0.8)

axes[1].set_title("RGB Kanal Histogramları")
axes[1].set_xlabel("Piksel Değeri (0-255)")
axes[1].set_ylabel("Frekans")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "histogram.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 5. İstatistiksel Analiz
# ============================================================
print("\n" + "=" * 60)
print("İSTATİSTİKSEL ANALİZ (Grayscale Görüntü)")
print("=" * 60)
print(f"  Min    : {np.min(img_gray)}")
print(f"  Max    : {np.max(img_gray)}")
print(f"  Median : {np.median(img_gray):.2f}")
print(f"  Mean   : {np.mean(img_gray):.2f}")
print(f"  Std    : {np.std(img_gray):.2f}")
print("=" * 60)

# Kanal bazlı istatistikler
print("\nKanal Bazlı İstatistikler (RGB):")
print(f"{'Kanal':<8} {'Min':<6} {'Max':<6} {'Median':<10} {'Mean':<10}")
print("-" * 40)
for i, name in enumerate(["Red", "Green", "Blue"]):
    channel = img_rgb[:, :, i]
    print(f"{name:<8} {np.min(channel):<6} {np.max(channel):<6} "
          f"{np.median(channel):<10.2f} {np.mean(channel):<10.2f}")

print("\n[OK] Uygulama 1 tamamlandı! Çıktılar 'outputs/' dizinine kaydedildi.")
