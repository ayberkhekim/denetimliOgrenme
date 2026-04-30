"""
Exercise 1: Goruntuden Belirli Bolge Cikarma Fonksiyonu
========================================================
Girdi : Herhangi bir sinifta (uint8, float32 vs.) goruntu + (x,y) piksel araliklari
Cikti : Verilen indislere kisitlanmis float32 matris + gorsellestirme

cameraman.tif uzerinde uygulanarak kameracinin basi cikarilir.
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_region(image_path, row_range, col_range):
    """
    Goruntuden belirli bir bolgeyi cikarir.

    Parametreler:
        image_path (str) : Goruntu dosya yolu
        row_range  (tuple): (baslangic_satir, bitis_satir) - satir araligi
        col_range  (tuple): (baslangic_sutun, bitis_sutun) - sutun araligi

    Donus:
        region (np.ndarray): float32 formatinda cikarilmis bolge matrisi
    """
    # Goruntuyu oku (grayscale)
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")

    # float32'ye donustur
    img_float = img.astype(np.float32)

    # Bolgeyi cikar
    r_start, r_end = row_range
    c_start, c_end = col_range
    region = img_float[r_start:r_end, c_start:c_end]

    # Gorsellestirme
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exercise 1: Bolge Cikarma", fontsize=14, fontweight="bold")

    axes[0].imshow(img_float, cmap="gray", interpolation="none")
    axes[0].set_title("Orijinal Goruntu")
    axes[0].axis("off")
    # Cikarilan bolgeyi kirmizi dikdortgenle goster
    from matplotlib.patches import Rectangle
    rect = Rectangle((c_start, r_start), c_end - c_start, r_end - r_start,
                      linewidth=2, edgecolor="red", facecolor="none")
    axes[0].add_patch(rect)

    axes[1].imshow(region, cmap="gray", interpolation="none")
    axes[1].set_title(f"Cikarilan Bolge [{r_start}:{r_end}, {c_start}:{c_end}]")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "exercise1_bolge_cikarma.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    return region


# ============================================================
# Uygulama: cameraman.tif'den kameracinin basini cikar
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
cameraman_path = os.path.join(script_dir, "cameraman.tif")

# Kameracinin bas bolgesi (yaklasik koordinatlar)
head_region = extract_region(cameraman_path,
                              row_range=(30, 120),
                              col_range=(80, 170))

print(f"Cikarilan bolge boyutu : {head_region.shape}")
print(f"Veri tipi              : {head_region.dtype}")
print(f"Min / Max              : {head_region.min():.1f} / {head_region.max():.1f}")
print("\n[OK] Exercise 1 tamamlandi!")
