"""
YZM0206 - Lab05 Uygulama 2: Uzamsal Filtreleme (Spatial Filtering / Convolution)
=================================================================================
Bu modülde lena.jpg görüntüsü üzerinde:
  - Gaussian Filtresi (3x3 ve 5x5)
  - Mean (Ortalama) Filtresi (3x3 ve 5x5)
  - Laplacian Filtresi (kenar tespiti)
  - Median Filtresi (gürültü temizleme)
  - Padding (sıfır ekleme) açıklaması
işlemleri gerçekleştirilmektedir.

Lineer Cebir Temeli:
  Convolution (gezdirme) işlemi, görüntü matrisinin her piksel komşuluğu ile
  bir kernel (filtre) matrisinin eleman-eleman çarpılıp toplanmasıdır:
    g(x,y) = Σ Σ h(i,j) · f(x-i, y-j)
  Bu, matris iç çarpımına (Frobenius inner product) karşılık gelir.
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

# Filtreleme işlemleri grayscale üzerinde yapılır
img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

print(f"Görüntü boyutu: {img_gray.shape}")

# ============================================================
# 1. Gaussian Filtresi (Yumuşatma / Blurring)
# ============================================================
# Gaussian kernel, merkeze daha fazla ağırlık veren bir ağırlıklı ortalamadır.
# Matematiksel olarak: h(x,y) = (1/2πsigma^2) · exp(-(x^2+y^2)/(2sigma^2))
# sigma değeri 0 verildiğinde OpenCV kernel boyutundan otomatik hesaplar.

gaussian_3x3 = cv.GaussianBlur(img_gray, (3, 3), 0)
gaussian_5x5 = cv.GaussianBlur(img_gray, (5, 5), 0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Gaussian Filtresi - Kernel Boyutu Karşılaştırması", fontsize=14, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Orijinal")
axes[0].axis("off")

axes[1].imshow(gaussian_3x3, cmap="gray")
axes[1].set_title("Gaussian 3×3")
axes[1].axis("off")

axes[2].imshow(gaussian_5x5, cmap="gray")
axes[2].set_title("Gaussian 5×5")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gaussian_filtre.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 2. Mean (Ortalama) Filtresi
# ============================================================
# Mean filtre kernel'i tüm elemanları eşit olan bir matristir:
# 3x3 için: h = (1/9) · [[1,1,1],[1,1,1],[1,1,1]]
# 5x5 için: h = (1/25) · ones(5,5)
# Bu, komşu piksellerin basit aritmetik ortalamasını alır.

mean_3x3 = cv.blur(img_gray, (3, 3))
mean_5x5 = cv.blur(img_gray, (5, 5))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Mean (Ortalama) Filtresi - Kernel Boyutu Karşılaştırması", fontsize=14, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Orijinal")
axes[0].axis("off")

axes[1].imshow(mean_3x3, cmap="gray")
axes[1].set_title("Mean 3×3")
axes[1].axis("off")

axes[2].imshow(mean_5x5, cmap="gray")
axes[2].set_title("Mean 5×5")
axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "mean_filtre.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 3. Gaussian vs Mean Karşılaştırması
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Gaussian vs Mean Filtresi Karşılaştırması", fontsize=14, fontweight="bold")

# Üst satır: 3x3
axes[0, 0].imshow(img_gray, cmap="gray")
axes[0, 0].set_title("Orijinal")
axes[0, 0].axis("off")

axes[0, 1].imshow(gaussian_3x3, cmap="gray")
axes[0, 1].set_title("Gaussian 3×3")
axes[0, 1].axis("off")

axes[0, 2].imshow(mean_3x3, cmap="gray")
axes[0, 2].set_title("Mean 3×3")
axes[0, 2].axis("off")

# Alt satır: 5x5
axes[1, 0].imshow(img_gray, cmap="gray")
axes[1, 0].set_title("Orijinal")
axes[1, 0].axis("off")

axes[1, 1].imshow(gaussian_5x5, cmap="gray")
axes[1, 1].set_title("Gaussian 5×5")
axes[1, 1].axis("off")

axes[1, 2].imshow(mean_5x5, cmap="gray")
axes[1, 2].set_title("Mean 5×5")
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "gaussian_vs_mean.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 4. Laplacian Filtresi (Kenar Tespiti)
# ============================================================
# Laplacian, ikinci türev operatörüdür: nabla^2f = d^2f/dx^2 + d^2f/dy^2
# Ayrık formda kernel:  [[0, 1, 0],
#                         [1,-4, 1],
#                         [0, 1, 0]]
# Yüksek frekans bileşenlerini (kenarlar) tespit eder.

laplacian = cv.Laplacian(img_gray, cv.CV_64F)
# Mutlak değer alıp uint8'e dönüştür (negatif değerler olabilir)
laplacian_abs = np.uint8(np.absolute(laplacian))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Laplacian Filtresi - Kenar Tespiti", fontsize=14, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Orijinal")
axes[0].axis("off")

axes[1].imshow(laplacian_abs, cmap="gray")
axes[1].set_title("Laplacian (Kenarlar)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "laplacian_filtre.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 5. Median Filtresi (Gürültü Temizleme)
# ============================================================
# Median filtre, lineer olmayan bir filtredir.
# Komşuluk içindeki piksel değerlerini sıralar ve ortanca değeri seçer.
# Salt-and-pepper (tuz-biber) gürültüsüne karşı çok etkilidir.

# Önce görüntüye salt-and-pepper gürültüsü ekleyelim
def add_salt_pepper_noise(image, amount=0.05):
    """Görüntüye tuz-biber gürültüsü ekler."""
    noisy = np.copy(image)
    # Tuz (beyaz piksel)
    num_salt = int(amount * image.size)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[salt_coords[0], salt_coords[1]] = 255
    # Biber (siyah piksel)
    num_pepper = int(amount * image.size)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy

np.random.seed(42)  # Tekrarlanabilirlik için
img_noisy = add_salt_pepper_noise(img_gray, amount=0.05)

# Median filtre uygula
median_3x3 = cv.medianBlur(img_noisy, 3)
median_5x5 = cv.medianBlur(img_noisy, 5)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("Median Filtresi - Gürültü Temizleme Performansı", fontsize=14, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title("Orijinal")
axes[0].axis("off")

axes[1].imshow(img_noisy, cmap="gray")
axes[1].set_title("Gürültülü (Salt & Pepper)")
axes[1].axis("off")

axes[2].imshow(median_3x3, cmap="gray")
axes[2].set_title("Median 3×3")
axes[2].axis("off")

axes[3].imshow(median_5x5, cmap="gray")
axes[3].set_title("Median 5×5")
axes[3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "median_filtre.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 6. Padding (Sıfır Ekleme) Açıklaması ve Gösterimi
# ============================================================
# Kenar piksellerinde filtre gezdirirken veri kaybını önlemek için
# görüntünün çevresine sıfır (0) değerli pikseller eklenir.
# Bu sayede çıkış görüntüsü, giriş görüntüsüyle aynı boyutta olur.
#
# Padding miktarı: p = (kernel_size - 1) / 2
# 3x3 kernel -> p = 1 piksel padding
# 5x5 kernel -> p = 2 piksel padding

# Manuel padding gösterimi
padded_img = cv.copyMakeBorder(img_gray, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Zero Padding Gösterimi", fontsize=14, fontweight="bold")

axes[0].imshow(img_gray, cmap="gray")
axes[0].set_title(f"Orijinal ({img_gray.shape[0]}×{img_gray.shape[1]})")
axes[0].axis("off")

axes[1].imshow(padded_img, cmap="gray")
axes[1].set_title(f"Padded ({padded_img.shape[0]}×{padded_img.shape[1]}, p=2)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "padding.png"), dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# 7. Teknik Karşılaştırma Raporu
# ============================================================
print("\n" + "=" * 65)
print("TEKNİK KARŞILAŞTIRMA: 3×3 vs 5×5 Filtre Boyutu")
print("=" * 65)
print("""
Filtre boyutu arttıkça:
  * Yumuşatma (blurring) etkisi artar -> görüntü daha bulanık olur
  * Gürültü temizleme performansı artar
  * Ancak kenar detayları (edge details) kaybolur
  * Hesaplama maliyeti artar: 3×3 = 9 çarpma, 5×5 = 25 çarpma

Gaussian vs Mean:
  * Gaussian: Merkeze yakın piksellere daha fazla ağırlık verir
    -> daha doğal bir yumuşatma sağlar
  * Mean: Tüm piksellere eşit ağırlık verir
    -> daha agresif bir yumuşatma, kenarları daha çok bulanıklaştırır

Laplacian:
  * İkinci türev operatörü olduğundan gürültüye hassastır
  * Yüksek frekans (kenarlar) bileşenlerini tespit eder
  * Genellikle öncesinde Gaussian yumuşatma uygulanır (LoG)

Median:
  * Lineer olmayan filtre -> impulse (salt-and-pepper) gürültüsüne
    karşı çok etkilidir
  * Kenarları koruyarak gürültüyü temizler
  * Mean filtresine göre kenar koruma performansı daha yüksektir
""")

print("[OK] Uygulama 2 tamamlandı! Çıktılar 'outputs/' dizinine kaydedildi.")
