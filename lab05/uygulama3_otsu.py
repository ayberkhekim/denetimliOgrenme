"""
YZM0206 - Lab05 Uygulama 3: Otsu'nun Eşikleme Metodu (Otsu's Thresholding)
============================================================================
Bu modülde:
  - Görüntüyü grayscale'e dönüştürme
  - Otsu's Thresholding ile otomatik eşik belirleme
  - Binary (ikili) görüntü oluşturma
işlemleri gerçekleştirilmektedir.

Otsu Algoritması:
  Otsu metodu, histogramı iki sınıfa ayıran optimal eşik değerini bulur.
  Sınıf-içi varyansı (within-class variance) minimize eden veya
  eşdeğer olarak sınıf-arası varyansı (between-class variance) maximize
  eden eşik değerini seçer:
    sigma^2_b(t) = omega₁(t)·omega₂(t)·[mu₁(t) - mu₂(t)]^2
  Bu tam bir arama (exhaustive search) olup t ∈ [0, 255] aralığında
  en iyi eşik değerini bulur.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 0. Hazırlık
# ============================================================
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

img_bgr = cv.imread("lena.jpg")
if img_bgr is None:
    raise FileNotFoundError("lena.jpg bulunamadı!")

# ============================================================
# 1. Grayscale Dönüşüm
# ============================================================
# RGB -> Grayscale dönüşüm formülü:
# Y = 0.299·R + 0.587·G + 0.114·B
# Bu, insan gözünün yeşile daha duyarlı olmasını yansıtır.
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

print(f"Grayscale görüntü boyutu: {img_gray.shape}")
print(f"Veri tipi: {img_gray.dtype}")

# ============================================================
# 2. Otsu's Thresholding
# ============================================================
# cv2.threshold parametreleri:
#   - src: giriş görüntüsü (grayscale)
#   - thresh: başlangıç eşik değeri (Otsu otomatik bulur, 0 verilir)
#   - maxval: eşiğin üstündeki piksellere atanacak değer (255)
#   - type: THRESH_BINARY + THRESH_OTSU
#
# Dönüş değerleri:
#   - otsu_thresh: Otsu'nun bulduğu optimal eşik değeri
#   - img_binary: ikili (binary) görüntü

otsu_thresh, img_binary = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

print(f"\nOtsu tarafından bulunan eşik değeri: {otsu_thresh}")

# ============================================================
# 3. Farklı Eşik Değerleri ile Karşılaştırma
# ============================================================
# Otsu'nun bulduğu değerin ne kadar iyi olduğunu görmek için
# farklı sabit eşik değerleri ile de karşılaştıralım.
_, binary_100 = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
_, binary_150 = cv.threshold(img_gray, 150, 255, cv.THRESH_BINARY)
_, binary_200 = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)

# ============================================================
# 4. Görselleştirme
# ============================================================
# Ana karşılaştırma: Orijinal vs Otsu Binary
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Otsu's Thresholding", fontsize=16, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Orijinal (Grayscale)")
axes[0].axis("off")

axes[1].imshow(img_binary, cmap="gray")
axes[1].set_title(f"Otsu Binary (Eşik = {otsu_thresh:.0f})")
axes[1].axis("off")

# Histogram + eşik çizgisi
axes[2].hist(img_gray.ravel(), bins=256, range=(0, 256), color="gray", alpha=0.7)
axes[2].axvline(x=otsu_thresh, color="red", linewidth=2, linestyle="--",
                label=f"Otsu Eşik = {otsu_thresh:.0f}")
axes[2].set_title("Histogram + Otsu Eşik Değeri")
axes[2].set_xlabel("Piksel Değeri")
axes[2].set_ylabel("Frekans")
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "otsu_esikleme.png"), dpi=150, bbox_inches="tight")
plt.show()

# Farklı eşik değerleri karşılaştırması
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Farklı Eşik Değerleri Karşılaştırması", fontsize=14, fontweight="bold")

thresholds = [
    (binary_100, "Eşik = 100"),
    (binary_150, "Eşik = 150"),
    (binary_200, "Eşik = 200"),
    (img_binary, f"Otsu Eşik = {otsu_thresh:.0f}"),
]

for ax, (img, title) in zip(axes, thresholds):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "esik_karsilastirma.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 5. Sonuç Raporu
# ============================================================
print("\n" + "=" * 60)
print("OTSU EŞİKLEME SONUÇLARI")
print("=" * 60)
print(f"  Otsu Eşik Değeri     : {otsu_thresh:.0f}")
print(f"  Beyaz piksel sayısı  : {np.sum(img_binary == 255)}")
print(f"  Siyah piksel sayısı  : {np.sum(img_binary == 0)}")
print(f"  Beyaz oranı          : {np.sum(img_binary == 255) / img_binary.size * 100:.1f}%")
print(f"  Siyah oranı          : {np.sum(img_binary == 0) / img_binary.size * 100:.1f}%")
print("=" * 60)

print("\n[OK] Uygulama 3 tamamlandı! Çıktılar 'outputs/' dizinine kaydedildi.")
