"""
YZM0206 - Lab05 Uygulama 4: RGB ve Grayscale Arasındaki İlişki
===============================================================
Bu modülde:
  - RGB görüntünün R, G, B kanallarını ayrı ayrı analiz etme
  - Grayscale dönüşüm formülünü açıklama ve doğrulama
  - Her kanalın grayscale'e katkısını görselleştirme
işlemleri gerçekleştirilmektedir.

Matematiksel Temel:
  Grayscale = 0.299·R + 0.587·G + 0.114·B

  Bu, matris formunda yazılabilir:
  Y = [0.299  0.587  0.114] · [R]
                                [G]
                                [B]

  Katsayılar, insan görsel algı sistemine (HVS) dayanır:
  - Yeşil kanalı en yüksek ağırlığa sahiptir (%58.7)
    çünkü insan gözü yeşile en duyarlıdır
  - Kırmızı (%29.9) ve Mavi (%11.4) daha düşük ağırlıktadır
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

img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

# ============================================================
# 1. Kanal Ayrımı
# ============================================================
# RGB görüntüsünden her bir kanalı ayrı ayrı çıkar
R = img_rgb[:, :, 0]  # Red kanalı
G = img_rgb[:, :, 1]  # Green kanalı
B = img_rgb[:, :, 2]  # Blue kanalı

# Her kanalı hem grayscale hem de renkli olarak gösterelim
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle("RGB Kanal Ayrımı", fontsize=16, fontweight="bold")

# Üst satır: Grayscale olarak kanallar
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Orijinal (RGB)")
axes[0, 0].axis("off")

axes[0, 1].imshow(R, cmap="gray")
axes[0, 1].set_title("R Kanalı (Grayscale)")
axes[0, 1].axis("off")

axes[0, 2].imshow(G, cmap="gray")
axes[0, 2].set_title("G Kanalı (Grayscale)")
axes[0, 2].axis("off")

axes[0, 3].imshow(B, cmap="gray")
axes[0, 3].set_title("B Kanalı (Grayscale)")
axes[0, 3].axis("off")

# Alt satır: Renkli olarak kanallar (diğer kanallar sıfırlanmış)
axes[1, 0].imshow(img_rgb)
axes[1, 0].set_title("Orijinal (RGB)")
axes[1, 0].axis("off")

# Sadece Red kanalı görünür
img_r_only = np.zeros_like(img_rgb)
img_r_only[:, :, 0] = R
axes[1, 1].imshow(img_r_only)
axes[1, 1].set_title("Sadece R Kanalı")
axes[1, 1].axis("off")

# Sadece Green kanalı görünür
img_g_only = np.zeros_like(img_rgb)
img_g_only[:, :, 1] = G
axes[1, 2].imshow(img_g_only)
axes[1, 2].set_title("Sadece G Kanalı")
axes[1, 2].axis("off")

# Sadece Blue kanalı görünür
img_b_only = np.zeros_like(img_rgb)
img_b_only[:, :, 2] = B
axes[1, 3].imshow(img_b_only)
axes[1, 3].set_title("Sadece B Kanalı")
axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kanal_ayrimi.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 2. Grayscale Dönüşüm Formülünün Doğrulanması
# ============================================================
# Manuel hesaplama: Y = 0.299·R + 0.587·G + 0.114·B
gray_manual = (0.299 * R.astype(np.float64) +
               0.587 * G.astype(np.float64) +
               0.114 * B.astype(np.float64))
gray_manual = gray_manual.astype(np.uint8)

# OpenCV ile dönüşüm
gray_opencv = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)

# Fark hesapla
diff = np.abs(gray_manual.astype(np.int16) - gray_opencv.astype(np.int16))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Grayscale Dönüşüm Doğrulaması", fontsize=14, fontweight="bold")

axes[0].imshow(gray_manual, cmap="gray")
axes[0].set_title("Manuel: 0.299R + 0.587G + 0.114B")
axes[0].axis("off")

axes[1].imshow(gray_opencv, cmap="gray")
axes[1].set_title("OpenCV: cv2.cvtColor(BGR2GRAY)")
axes[1].axis("off")

axes[2].imshow(diff, cmap="hot")
axes[2].set_title(f"Fark (Max: {diff.max()}, Mean: {diff.mean():.3f})")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "grayscale_dogrulama.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 3. Her Kanalın Grayscale'e Katkısı
# ============================================================
# Weighted katkıları hesapla
R_contribution = (0.299 * R.astype(np.float64)).astype(np.uint8)
G_contribution = (0.587 * G.astype(np.float64)).astype(np.uint8)
B_contribution = (0.114 * B.astype(np.float64)).astype(np.uint8)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Her Kanalın Grayscale'e Ağırlıklı Katkısı", fontsize=14, fontweight="bold")

axes[0].imshow(gray_opencv, cmap="gray")
axes[0].set_title("Sonuç: Grayscale")
axes[0].axis("off")

axes[1].imshow(R_contribution, cmap="gray")
axes[1].set_title("R × 0.299 (%29.9)")
axes[1].axis("off")

axes[2].imshow(G_contribution, cmap="gray")
axes[2].set_title("G × 0.587 (%58.7)")
axes[2].axis("off")

axes[3].imshow(B_contribution, cmap="gray")
axes[3].set_title("B × 0.114 (%11.4)")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kanal_katkilari.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 4. Kanal İstatistikleri
# ============================================================
print("\n" + "=" * 65)
print("RGB KANAL İSTATİSTİKLERİ")
print("=" * 65)
print(f"{'Kanal':<12} {'Min':<6} {'Max':<6} {'Mean':<10} {'Median':<10} {'Std':<10}")
print("-" * 54)

for name, channel in [("Red (R)", R), ("Green (G)", G), ("Blue (B)", B),
                       ("Grayscale", gray_opencv)]:
    print(f"{name:<12} {np.min(channel):<6} {np.max(channel):<6} "
          f"{np.mean(channel):<10.2f} {np.median(channel):<10.2f} {np.std(channel):<10.2f}")

print("\n" + "=" * 65)
print("GRAYSCALE DÖNÜŞÜM DOĞRULAMASI")
print("=" * 65)
print(f"  Manuel vs OpenCV maksimum fark : {diff.max()}")
print(f"  Manuel vs OpenCV ortalama fark : {diff.mean():.4f}")
print(f"  Formül: Gray = 0.299·R + 0.587·G + 0.114·B")
print(f"  -> Yeşil kanalı en yüksek ağırlığa sahiptir (%58.7)")
print(f"  -> İnsan gözü yeşile en duyarlıdır (cone hücreleri)")
print("=" * 65)

print("\n[OK] Uygulama 4 tamamlandı! Çıktılar 'outputs/' dizinine kaydedildi.")
