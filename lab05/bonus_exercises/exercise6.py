"""
Exercise 6: Pecete Deseni (Napkin Pattern)
===========================================
10x10 piksellik karelerden olusan pecete deseni olusturur.
numpy.tile komutu kullanilarak temel desen tekrarlanir.

Pecete deseni, satranc tahtasina benzer ancak daha kucuk
kareler ve daha yogun bir tekrar iceren bir desendir.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Pecete Deseni Olustur
# ============================================================
square_size = 10  # Her karenin boyutu: 10x10 piksel

# Temel 2x2 desen (siyah ve beyaz)
base = np.array([[255, 0],
                  [0, 255]], dtype=np.uint8)

# Her pikseli 10x10 kareye buyut
base_block = np.kron(base, np.ones((square_size, square_size), dtype=np.uint8))

# Deseni tekrarla -> buyuk pecete deseni (ornegin 500x500)
# base_block = 20x20 piksel, 25 kez tekrarlayinca 500x500 olur
repeat_count = 25
napkin = np.tile(base_block, (repeat_count, repeat_count))

print(f"Temel blok boyutu  : {base_block.shape}")
print(f"Pecete boyutu      : {napkin.shape}")
print(f"Kare boyutu        : {square_size}x{square_size} piksel")

# ============================================================
# 2. Gorsellestirme
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Exercise 6: Pecete Deseni (10x10 kareler)",
             fontsize=14, fontweight="bold")

# Yakin goruntu (detay)
axes[0].imshow(napkin[:100, :100], cmap="gray", interpolation="none")
axes[0].set_title("Yakin Gorunum (100x100 piksel)")
axes[0].axis("off")

# Tam goruntu
axes[1].imshow(napkin, cmap="gray", interpolation="none")
axes[1].set_title(f"Tam Pecete Deseni ({napkin.shape[0]}x{napkin.shape[1]})")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise6_pecete.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Dosyaya kaydet
cv.imwrite(os.path.join(OUTPUT_DIR, "napkin_pattern.png"), napkin)

print(f"\nKullanilan yontem:")
print(f"  1. 2x2 temel desen: [[255, 0], [0, 255]]")
print(f"  2. np.kron(base, ones(10,10)) -> 20x20 blok")
print(f"  3. np.tile(block, (25, 25)) -> 500x500 pecete")
print(f"\n[OK] Exercise 6 tamamlandi!")
