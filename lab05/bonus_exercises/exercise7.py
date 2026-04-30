"""
Exercise 7: Noktalar Deseni (Dots Pattern)
============================================
500x500 piksellik goruntu uzerinde duzgun aralikli daireler olusturur.
Her dairenin yaricapi 10 piksel, merkezler arasi mesafe 50 pikseldir.

Yontem:
  - 50x50 piksellik temel hucre icinde merkezi bir daire olustur
  - np.tile ile deseni tekrarla
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Temel Hucre Olustur
# ============================================================
cell_size = 50   # Merkezler arasi mesafe
radius = 10      # Daire yaricapi
img_size = 500   # Sonuc goruntu boyutu

# 50x50 piksellik temel hucre
cell = np.zeros((cell_size, cell_size), dtype=np.uint8)

# Hucrenin merkezinde yaricapi 10 olan daire
cx, cy = cell_size // 2, cell_size // 2
y, x = np.ogrid[:cell_size, :cell_size]
circle_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
cell[circle_mask] = 255

# ============================================================
# 2. Deseni Tekrarla
# ============================================================
# 500 / 50 = 10 tekrar her yonde
repeat = img_size // cell_size
dots_pattern = np.tile(cell, (repeat, repeat))

print(f"Temel hucre boyutu   : {cell.shape}")
print(f"Daire yaricapi       : {radius} piksel")
print(f"Merkezler arasi      : {cell_size} piksel")
print(f"Sonuc goruntu boyutu : {dots_pattern.shape}")
print(f"Toplam daire sayisi  : {repeat * repeat}")

# ============================================================
# 3. Gorsellestirme
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Exercise 7: Noktalar Deseni (r=10, aralik=50)",
             fontsize=14, fontweight="bold")

# Temel hucre
axes[0].imshow(cell, cmap="gray", interpolation="none")
axes[0].set_title(f"Temel Hucre ({cell_size}x{cell_size})")
axes[0].axis("off")

# Tam desen
axes[1].imshow(dots_pattern, cmap="gray", interpolation="none")
axes[1].set_title(f"Tam Desen ({img_size}x{img_size})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise7_noktalar.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Dosyaya kaydet
cv.imwrite(os.path.join(OUTPUT_DIR, "dots_pattern.png"), dots_pattern)

print(f"\nKullanilan yontem:")
print(f"  1. 50x50 hucre icinde r=10 daire olustur")
print(f"  2. np.tile(cell, (10, 10)) -> 500x500 desen")
print(f"\n[OK] Exercise 7 tamamlandi!")
