"""
Exercise 4: Satranc Tahtasi (Chess Checkerboard)
==================================================
250x250 piksellik karelerden olusan satranc tahtasi olusturur.
numpy.tile komutu kullanilarak temel 2x2 desen tekrarlanir.

Yontem:
  - 2x2'lik temel desen: [[beyaz, siyah], [siyah, beyaz]]
  - Her hucre 250x250 piksel -> np.kron ile olcekleme
  - Alternatif: np.tile ile dogrudan tekrar
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. Satranc Tahtasi Olustur
# ============================================================
square_size = 250  # Her karenin boyutu (piksel)
board_squares = 8  # 8x8 satranc tahtasi

# Yontem: np.tile ile
# Temel 2x2 desen (siyah=0, beyaz=255)
base_pattern = np.array([[255, 0],
                          [0, 255]], dtype=np.uint8)

# Temel deseni 4x4 kez tekrarla -> 8x8 kare
tiled = np.tile(base_pattern, (board_squares // 2, board_squares // 2))

# Her pikseli 250x250'lik kareye buyut (Kronecker carpimi)
# np.kron: Kronecker product -> her elemani bir blok matris ile carpar
chess_board = np.kron(tiled, np.ones((square_size, square_size), dtype=np.uint8))

print(f"Satranc tahtasi boyutu: {chess_board.shape}")
print(f"  = {board_squares}x{board_squares} kare x {square_size}x{square_size} piksel")

# ============================================================
# 2. Gorsellestirme
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
fig.suptitle("Exercise 4: Satranc Tahtasi (250x250 kareler)",
             fontsize=14, fontweight="bold")

ax.imshow(chess_board, cmap="gray", interpolation="none")
ax.set_title(f"Boyut: {chess_board.shape[0]}x{chess_board.shape[1]}")
ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exercise4_satranc.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Dosyaya kaydet
cv.imwrite(os.path.join(OUTPUT_DIR, "chess_board.png"), chess_board)

print(f"\nKullanilan yontem:")
print(f"  1. 2x2 temel desen: [[255, 0], [0, 255]]")
print(f"  2. np.tile(pattern, (4, 4)) -> 8x8 kare")
print(f"  3. np.kron(tiled, ones(250,250)) -> piksellere buyutme")
print(f"\n[OK] Exercise 4 tamamlandi!")
