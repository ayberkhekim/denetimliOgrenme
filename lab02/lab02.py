import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. VERİ YÜKLEME VE ÖN İŞLEME
# ==========================================
file_path = r'C:\Users\ayberk\Desktop\Lab02\CarSales.xlsx'

if not os.path.exists(file_path):
    print(f"HATA: Dosya bulunamadı: {file_path}")
    exit()

# Dosyayı oku ve sütun isimlerini temizle
df = pd.read_excel(file_path)
df.columns = df.columns.str.strip()

# Dosyandaki gerçek sütun isimlerini kullanıyoruz
target = 'Price_in_thousands'
feat1 = 'Horsepower'
feat2 = 'Engine_size'

# Boş verileri temizle (Hesaplamaların bozulmaması için kritik)
df = df.dropna(subset=[target, feat1, feat2])

# Veriyi normalleştir (Gradyan İnişi için 0-1 arasına çekme)
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[[target, feat1, feat2]]),
                         columns=[target, feat1, feat2])

# ==========================================
# 2. BÖLÜM A: BASİT DOĞRUSAL REGRESYON
# ==========================================
X_a = df_scaled[feat1].values
y = df_scaled[target].values
n = len(y)

w, b = 0.0, 0.0
lr = 0.1
epochs = 150
a_mse_hist, a_w_hist, a_b_hist = [], [], []

for _ in range(epochs):
    y_pred = w * X_a + b
    error = y_pred - y
    grad_w = (2/n) * np.sum(error * X_a)
    grad_b = (2/n) * np.sum(error)
    w -= lr * grad_w
    b -= lr * grad_b
    a_mse_hist.append(np.mean(error**2))
    a_w_hist.append(w)
    a_b_hist.append(b)

# ==========================================
# 3. BÖLÜM B: ÇOKLU DOĞRUSAL REGRESYON
# ==========================================
X_b = df_scaled[[feat1, feat2]].values
W_multi = np.zeros(2)
b_multi = 0.0
b_mse_hist = []

for _ in range(epochs):
    y_pred = np.dot(X_b, W_multi) + b_multi
    error = y_pred - y
    grad_W = (2/n) * np.dot(X_b.T, error)
    grad_b_m = (2/n) * np.sum(error)
    W_multi -= lr * grad_W
    b_multi -= lr * grad_b_m
    b_mse_hist.append(np.mean(error**2))

# ==========================================
# 4. GÖRSELLEŞTİRME (RAPOR İÇİN)
# ==========================================

# --- ŞEKİL 1: BÖLÜM A GRAFİKLERİ ---
fig1 = plt.figure(figsize=(15, 5))
fig1.suptitle('Bölüm A: Basit Doğrusal Regresyon Analizi', fontsize=16)

# 1.1. Regresyon Doğrusu
ax1 = fig1.add_subplot(131)
ax1.scatter(X_a, y, alpha=0.5, label='Veri')
ax1.plot(X_a, w * X_a + b, color='red', label='Model')
ax1.set_xlabel(feat1)
ax1.set_ylabel('Fiyat')
ax1.legend()

# 1.2. 3D Hata Yüzeyi (Loss Surface)
ax2 = fig1.add_subplot(132, projection='3d')
ws = np.linspace(w-1, w+1, 50)
bs = np.linspace(b-1, b+1, 50)
WS, BS = np.meshgrid(ws, bs)
ZS = np.array([np.mean((wi*X_a + bi - y)**2) for wi, bi in zip(np.ravel(WS), np.ravel(BS))]).reshape(WS.shape)
ax2.plot_surface(WS, BS, ZS, cmap='viridis', alpha=0.6)
ax2.plot(a_w_hist, a_b_hist, a_mse_hist, color='red', marker='o', markersize=2)
ax2.set_title('Gradyan İnişi Yolu (3D)')

# 1.3. MSE Düşüşü
ax3 = fig1.add_subplot(133)
ax3.plot(a_mse_hist, color='green')
ax3.set_title('Hata Düşüşü (MSE)')
ax3.set_xlabel('Epoch')

# --- ŞEKİL 2: BÖLÜM B GRAFİKLERİ ---
fig2 = plt.figure(figsize=(12, 6))
fig2.suptitle('Bölüm B: Çoklu Doğrusal Regresyon Analizi', fontsize=16)

# 2.1. 3D Veri ve Regresyon Düzlemi
ax4 = fig2.add_subplot(121, projection='3d')
ax4.scatter(df_scaled[feat1], df_scaled[feat2], y, color='blue', alpha=0.4)

# Düzlemi oluştur
x_range = np.linspace(0, 1, 10)
y_range = np.linspace(0, 1, 10)
XX, YY = np.meshgrid(x_range, y_range)
ZZ = W_multi[0]*XX + W_multi[1]*YY + b_multi
ax4.plot_surface(XX, YY, ZZ, color='red', alpha=0.3)
ax4.set_xlabel(feat1)
ax4.set_ylabel(feat2)
ax4.set_zlabel('Fiyat')
ax4.set_title('Çoklu Regresyon Düzlemi')

# 2.2. MSE Karşılaştırma
ax5 = fig2.add_subplot(122)
ax5.plot(a_mse_hist, label='Basit Regresyon MSE', linestyle='--')
ax5.plot(b_mse_hist, label='Çoklu Regresyon MSE', linewidth=2)
ax5.set_title('Basit vs Çoklu Hata Karşılaştırması')
ax5.set_xlabel('Epoch')
ax5.legend()

plt.tight_layout()
plt.show()

print(f"Basit Model: Price = {w:.2f}*{feat1} + {b:.2f}")
print(f"Çoklu Model: Price = {W_multi[0]:.2f}*{feat1} + {W_multi[1]:.2f}*{feat2} + {b_multi:.2f}")