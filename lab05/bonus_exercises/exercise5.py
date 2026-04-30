"""
Exercise 5: Es Merkezli Daireler (Concentric Circles)
======================================================
500x500 piksellik goruntu uzerinde es merkezli daireler olusturur.
Her daire cizgisi yaklasik 10 piksel genisligindedir.

Yontem:
  - Merkez: (250, 250)
  - Her piksel icin merkeze uzaklik hesaplanir
  - Uzaklik degeri periyodik olarak modulo alinir
  - 10 piksellik bantlar halinde siyah-beyaz gecis saglanir
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Es Merkezli Daireler Olustur
# ============================================================
size = 500
line_width = 10  # Her daire cizgisinin genisligi (piksel)

# Merkez
cx, cy = size // 2, size // 2

# Her piksel icin merkeze Oklidyen uzaklik hesapla
y, x = np.ogrid[:size, :size]
distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

# Periyodik desen: uzaklik degerini (2 * line_width) ile modulo al
# line_width'den kucukse beyaz, degilse siyah
# Bu, 10px beyaz + 10px siyah seklinde bantlar olusturur
period = 2 * line_width
circles = np.zeros((size, size), dtype=np.uint8)
circles[((distance % period) < line_width)] = 255

print(f"Goruntu boyutu    : {circles.shape}")
print(f"Daire genisligi   : {line_width} piksel")
print(f"Periyot           : {period} piksel")

# ============================================================
# 2. Gorsellestirme
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle("Exercise 5: Es Merkezli Daireler (500x500)",
             fontsize=14, fontweight="bold")

ax.imshow(circles, cmap="gray", interpolation="none")
ax.set_title(f"Cizgi genisligi: ~{line_width}px")
ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise5_daireler.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Dosyaya kaydet
cv.imwrite(os.path.join(OUTPUT_DIR, "concentric_circles.png"), circles)

print(f"\n[OK] Exercise 5 tamamlandi!")
