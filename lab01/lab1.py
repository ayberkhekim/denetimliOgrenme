import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. VERİ SETİNİ YÜKLEME ---
file_path = r'C:\Users\ayberk\Desktop\CarSales.xlsx'

try:
    # Veriyi yükle ve eksik verileri temizle [cite: 4]
    df = pd.read_excel(file_path).dropna()
    df.columns = df.columns.str.strip()
    print("Excel başarıyla okundu.")
except Exception as e:
    print(f"Dosya okuma hatası: {e}")
    exit()

# Sütun isimlerini tanımla [cite: 2, 6]
y = df['Price_in_thousands']
X_s = df[['Horsepower']]

# --- A. BASİT DOĞRUSAL REGRESYON --- [cite: 5, 6]
model_s = LinearRegression().fit(X_s, y)
y_pred_s = model_s.predict(X_s)
mse_s = mean_squared_error(y, y_pred_s)
r2_s = r2_score(y, y_pred_s)

# --- B. ÇOKLU DOĞRUSAL REGRESYON --- [cite: 8, 9]
X_m = df[['Horsepower', 'Engine_size', 'Curb_weight']]
model_m = LinearRegression().fit(X_m, y)
y_pred_m = model_m.predict(X_m)
mse_m = mean_squared_error(y, y_pred_m)
r2_m = r2_score(y, y_pred_m)

# --- C. POLİNOMİYAL REGRESYON (Degree=3) --- [cite: 11, 12]
poly = PolynomialFeatures(degree=3)
# PolynomialFeatures'ı isimlerle eğitmek için DataFrame veriyoruz
X_p_feat = poly.fit_transform(X_s)
# Uyarıyı önlemek için özellikleri isimlendiriyoruz
poly_cols = poly.get_feature_names_out(['Horsepower'])
X_p_df = pd.DataFrame(X_p_feat, columns=poly_cols)

model_p = LinearRegression().fit(X_p_df, y)
y_pred_p = model_p.predict(X_p_df)
mse_p = mean_squared_error(y, y_pred_p)
r2_p = r2_score(y, y_pred_p)

# --- D. RIDGE REGRESYON (Alpha=10) --- [cite: 14, 15]
model_r = Ridge(alpha=10).fit(X_s, y)
y_pred_r = model_r.predict(X_s)
mse_r = mean_squared_error(y, y_pred_r)
r2_r = r2_score(y, y_pred_r)

# --- GÖRSELLEŞTİRME --- [cite: 7, 10, 13, 16]
plt.figure(figsize=(15, 10))

# Grafik A: Simple
plt.subplot(2, 2, 1)
plt.scatter(X_s, y, color='blue', s=10, alpha=0.5)
plt.plot(X_s, y_pred_s, color='red')
plt.title('A: Simple Linear Regression')

# Grafik B: 3D Görünüm (Multiple için Horsepower ve Engine_size baz alınmıştır)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 2, 2, projection='3d')
ax.scatter(df['Horsepower'], df['Engine_size'], y, c='blue', alpha=0.5)
plt.title('B: Multiple Regression (3D View)')

# Grafik C: Polynomial
plt.subplot(2, 2, 3)
X_range = np.linspace(X_s.min(), X_s.max(), 100).reshape(-1, 1)
X_range_df = pd.DataFrame(X_range, columns=['Horsepower'])
X_range_poly = pd.DataFrame(poly.transform(X_range_df), columns=poly_cols)
plt.scatter(X_s, y, color='blue', s=10, alpha=0.5)
plt.plot(X_range, model_p.predict(X_range_poly), color='green')
plt.title('C: Polynomial Regression (Deg 3)')

# Grafik D: Ridge
plt.subplot(2, 2, 4)
plt.scatter(X_s, y, color='blue', s=10, alpha=0.5)
plt.plot(X_s, y_pred_r, color='orange')
plt.title('D: Ridge Regression (Alpha=10)')

plt.tight_layout()
plt.show()

# --- SONUÇ TABLOSU --- [cite: 18]
print("\n" + "="*50)
print("3. SONUÇLARIN KIYASLANMASI")
print("="*50)
print(f"{'Model Türü':<20} | {'MSE (Hata)':<12} | {'R-Squared':<10}")
print("-" * 50)
print(f"{'Simple Linear':<20} | {mse_s:12.2f} | {r2_s:.4f}")
print(f"{'Multiple Linear':<20} | {mse_m:12.2f} | {r2_m:.4f}")
print(f"{'Polynomial (Deg:3)':<20} | {mse_p:12.2f} | {r2_p:.4f}")
print(f"{'Ridge Regression':<20} | {mse_r:12.2f} | {r2_r:.4f}")