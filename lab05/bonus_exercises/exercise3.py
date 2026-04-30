"""
Exercise 3: Lineer Degradasyon (Linear Degradation)
====================================================
Goruntuyu dikey yonde kademeli olarak karartma efekti uygular.

Yontem:
  - np.linspace(1, 0, rows) ile 1'den 0'a azalan bir vektor olusturulur
  - np.tile ile bu vektor sutun sayisi kadar tekrarlanarak maske matrisi elde edilir
  - Goruntu * maske = kademeli karartilmis goruntu

Lineer cebir perspektifinden bu, bir diagonal-benzeri matris ile
element-wise carpimdir (Hadamard product).
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
img = cv.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lena.jpg"),
                cv.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("lena.jpg bulunamadi!")

img_float = img.astype(np.float64)
rows, cols = img_float.shape

# ============================================================
# 2. Degradasyon Maskesi Olustur
# ============================================================
# Dikey degradasyon: ust satirdan (1.0) alt satira (0.0) dogru azalir
# np.linspace ile 1'den 0'a azalan vektor
gradient_vector = np.linspace(1, 0, rows)

# np.tile ile bu vektoru sutun sayisi kadar tekrarla
# Sonuc: (rows, cols) boyutunda maske matrisi
# Her satirdaki tum sutunlar ayni deger alir
degradation_mask = np.tile(gradient_vector.reshape(-1, 1), (1, cols))

# Yatay degradasyon da deneyelim
gradient_h = np.linspace(1, 0, cols)
degradation_mask_h = np.tile(gradient_h.reshape(1, -1), (rows, 1))

# ============================================================
# 3. Degradasyonu Uygula
# ============================================================
degraded_vertical = img_float * degradation_mask
degraded_horizontal = img_float * degradation_mask_h

# ============================================================
# 4. Gorsellestirme
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Exercise 3: Lineer Degradasyon", fontsize=16, fontweight="bold")

axes[0, 0].imshow(img_float, cmap="gray")
axes[0, 0].set_title("Orijinal")
axes[0, 0].axis("off")

axes[0, 1].imshow(degradation_mask, cmap="gray")
axes[0, 1].set_title("Dikey Degradasyon Maskesi")
axes[0, 1].axis("off")

axes[1, 0].imshow(degraded_vertical, cmap="gray")
axes[1, 0].set_title("Dikey Degradasyon Uygulanmis")
axes[1, 0].axis("off")

axes[1, 1].imshow(degraded_horizontal, cmap="gray")
axes[1, 1].set_title("Yatay Degradasyon Uygulanmis")
axes[1, 1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise3_degradasyon.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Sonucu kaydet
result = degraded_vertical.astype(np.uint8)
cv.imwrite(os.path.join(OUTPUT_DIR, "lena_degraded.jpg"), result)

print("Degradasyon maskesi olusturma yontemi:")
print("  1. np.linspace(1, 0, rows) -> 1'den 0'a azalan vektor")
print("  2. np.tile(..., (1, cols)) -> satir vektorunu matrise genislet")
print("  3. img * mask -> Hadamard (element-wise) carpim")
print("\n[OK] Exercise 3 tamamlandi!")
