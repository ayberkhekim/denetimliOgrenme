"""
Exercise 2: Dairesel Maske ile Bolge Cikarma
=============================================
lena_gray_512.tif uzerinde yaricapi 150 piksel olan dairesel bir maske
olusturularak goruntunun belirli bir bolgesi cikarilir.

Adimlar:
  1. Goruntuyu oku ve float64'e donustur
  2. Ayni boyutta sifir matrisi olustur
  3. (j - cx)^2 + (i - cy)^2 < 150^2 kosulunu saglayan pikselleri 1 yap
  4. Goruntu ile maskeyi element-wise carp (Hadamard carpimi)
  5. Daire disi pikselleri yarim yogunlukla goster
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Goruntuyu Oku
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
img = cv.imread(os.path.join(script_dir, "lena_gray_512.tif"), cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("lena_gray_512.tif bulunamadi!")

img_float = img.astype(np.float64)
rows, cols = img_float.shape
print(f"Goruntu boyutu: {rows} x {cols}")

# ============================================================
# 2. Dairesel Maske Olustur
# ============================================================
# Merkez koordinatlari
cx, cy = cols // 2, rows // 2
radius = 150

# Sifir matrisi
mask = np.zeros((rows, cols), dtype=np.float64)

# Dairesel bolgeyi 1 yap
# Vektorize versiyon (dongulerden cok daha hizli):
y_indices, x_indices = np.ogrid[:rows, :cols]
circle_condition = (x_indices - cx) ** 2 + (y_indices - cy) ** 2 < radius ** 2
mask[circle_condition] = 1.0

# ============================================================
# 3. Maskeyi Uygula (Hadamard / element-wise carpim)
# ============================================================
# Daire ici: orijinal yogunluk
masked_image = img_float * mask

# ============================================================
# 4. Daire Disi Yarim Yogunluk
# ============================================================
# Daire disindaki pikselleri yarim yogunlukla goster
half_mask = np.copy(mask)
half_mask[~circle_condition] = 0.5  # daire disi yarim yogunluk
masked_half = img_float * half_mask

# ============================================================
# 5. Gorsellestirme
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Exercise 2: Dairesel Maske (r=150)", fontsize=16, fontweight="bold")

axes[0, 0].imshow(img_float, cmap="gray")
axes[0, 0].set_title("Orijinal")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask, cmap="gray")
axes[0, 1].set_title(f"Maske (r={radius}, merkez=({cx},{cy}))")
axes[0, 1].axis("off")

axes[1, 0].imshow(masked_image, cmap="gray")
axes[1, 0].set_title("Maskelenmis (Daire disi siyah)")
axes[1, 0].axis("off")

axes[1, 1].imshow(masked_half, cmap="gray")
axes[1, 1].set_title("Maskelenmis (Daire disi yarim yogunluk)")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise2_dairesel_maske.png"),
            dpi=150, bbox_inches="tight")
plt.show()

print(f"Merkez          : ({cx}, {cy})")
print(f"Yaricap         : {radius}")
print(f"Daire ici piksel: {np.sum(circle_condition)}")
print(f"Daire disi      : {rows * cols - np.sum(circle_condition)}")
print("\n[OK] Exercise 2 tamamlandi!")
