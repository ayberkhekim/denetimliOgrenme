"""
YZM0206 LAB 7 - KNN (K-Nearest Neighbors) ile Müşteri Segmentasyonu
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import os

OUT = r"c:\Users\ayberk\Desktop\lab07\outputs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# ADIM 1 - Veri Yükleme ve Keşifsel Analiz (EDA)
# ============================================================
print("="*60)
print("ADIM 1: VERİ YÜKLEME VE KEŞİFSEL ANALİZ")
print("="*60)

df = pd.read_csv(r"c:\Users\ayberk\Desktop\lab07\teleCust1000t.csv")
print("\n--- df.head() ---")
print(df.head())
print("\n--- df.shape ---")
print(f"Satır: {df.shape[0]}, Sütun: {df.shape[1]}")
print("\n--- df.info() ---")
print(df.dtypes)
print("\n--- df.describe() ---")
print(df.describe().round(2))

# Sınıf dağılımı
print("\n--- Sınıf Dağılımı (custcat) ---")
sinif_dagilimi = df['custcat'].value_counts().sort_index()
sinif_isimleri = {1: 'Basic Service', 2: 'E-Service', 3: 'Plus Service', 4: 'Total Service'}
for k, v in sinif_dagilimi.items():
    print(f"  {k} ({sinif_isimleri[k]}): {v} müşteri ({v/len(df)*100:.1f}%)")

# Sınıf dağılımı grafiği
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
bars = ax.bar([sinif_isimleri[i] for i in sinif_dagilimi.index], sinif_dagilimi.values, color=colors, edgecolor='black')
for bar, val in zip(bars, sinif_dagilimi.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', fontweight='bold', fontsize=12)
ax.set_title('Müşteri Kategorisi Dağılımı (custcat)', fontsize=14, fontweight='bold')
ax.set_xlabel('Kategori')
ax.set_ylabel('Müşteri Sayısı')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '01_sinif_dagilimi.png'), dpi=150)
plt.close()
print("\n[OK] Sınıf dağılımı grafiği kaydedildi.")

# ============================================================
# ADIM 2 - Veri Hazırlama
# ============================================================
print("\n" + "="*60)
print("ADIM 2: VERİ HAZIRLAMA")
print("="*60)

X = df.drop('custcat', axis=1).values
y = df['custcat'].values
print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X type: {type(X)}, y type: {type(y)}")

# Normalizasyon
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print("\nNormalizasyon öncesi (ilk satır):", X[0][:5].round(2))
print("Normalizasyon sonrası (ilk satır):", X_norm[0][:5].round(2))

# ============================================================
# ADIM 3 - Train/Test Bölme
# ============================================================
print("\n" + "="*60)
print("ADIM 3: TRAIN/TEST BÖLME")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_norm, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {X_train.shape[0]} örnek")
print(f"Test:  {X_test.shape[0]} örnek")

# ============================================================
# ADIM 4 - K=4 ile KNN Modeli
# ============================================================
print("\n" + "="*60)
print("ADIM 4: K=4 İLE KNN MODELİ")
print("="*60)

knn4 = KNeighborsClassifier(n_neighbors=4)
knn4.fit(X_train, y_train)
y_pred_4 = knn4.predict(X_test)

acc4 = accuracy_score(y_test, y_pred_4)
f1_4 = f1_score(y_test, y_pred_4, average='weighted')
print(f"K=4 Accuracy: {acc4:.4f}")
print(f"K=4 F1-Score (weighted): {f1_4:.4f}")
print("\nClassification Report (K=4):")
print(classification_report(y_test, y_pred_4, target_names=list(sinif_isimleri.values())))

# Confusion Matrix K=4
fig, ax = plt.subplots(figsize=(7, 6))
cm4 = confusion_matrix(y_test, y_pred_4)
disp4 = ConfusionMatrixDisplay(cm4, display_labels=list(sinif_isimleri.values()))
disp4.plot(ax=ax, cmap='Blues', colorbar=True)
ax.set_title('Confusion Matrix (K=4)', fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '02_confusion_matrix_k4.png'), dpi=150)
plt.close()
print("[OK] K=4 Confusion Matrix kaydedildi.")

# ============================================================
# ADIM 5 - K=6 ile Karşılaştırma
# ============================================================
print("\n" + "="*60)
print("ADIM 5: K=6 İLE KARŞILAŞTIRMA")
print("="*60)

knn6 = KNeighborsClassifier(n_neighbors=6)
knn6.fit(X_train, y_train)
y_pred_6 = knn6.predict(X_test)

acc6 = accuracy_score(y_test, y_pred_6)
f1_6 = f1_score(y_test, y_pred_6, average='weighted')
print(f"K=6 Accuracy: {acc6:.4f}")
print(f"K=6 F1-Score (weighted): {f1_6:.4f}")
print("\nClassification Report (K=6):")
print(classification_report(y_test, y_pred_6, target_names=list(sinif_isimleri.values())))

# Confusion Matrix K=6
fig, ax = plt.subplots(figsize=(7, 6))
cm6 = confusion_matrix(y_test, y_pred_6)
disp6 = ConfusionMatrixDisplay(cm6, display_labels=list(sinif_isimleri.values()))
disp6.plot(ax=ax, cmap='Oranges', colorbar=True)
ax.set_title('Confusion Matrix (K=6)', fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '03_confusion_matrix_k6.png'), dpi=150)
plt.close()

print(f"\n--- K=4 vs K=6 Karşılaştırma ---")
print(f"  K=4 → Accuracy: {acc4:.4f}, F1: {f1_4:.4f}")
print(f"  K=6 → Accuracy: {acc6:.4f}, F1: {f1_6:.4f}")
better = 4 if f1_4 > f1_6 else 6
print(f"  >> K={better} daha iyi performans gösterdi.")

# ============================================================
# ADIM 6 - Optimum K Değerini Bulma (K=1..20)
# ============================================================
print("\n" + "="*60)
print("ADIM 6: OPTİMUM K DEĞERİNİ BULMA")
print("="*60)

k_range = range(1, 21)
acc_list = []
f1_list = []
train_acc_list = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    yp = knn_temp.predict(X_test)
    yp_train = knn_temp.predict(X_train)
    acc_list.append(accuracy_score(y_test, yp))
    f1_list.append(f1_score(y_test, yp, average='weighted'))
    train_acc_list.append(accuracy_score(y_train, yp_train))
    print(f"  K={k:2d} → Test Acc: {acc_list[-1]:.4f}, Train Acc: {train_acc_list[-1]:.4f}, F1: {f1_list[-1]:.4f}")

best_k = list(k_range)[np.argmax(f1_list)]
print(f"\n>> Optimum K = {best_k} (F1-Score: {max(f1_list):.4f})")

# Grafik: K vs Accuracy + F1
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, acc_list, 'b-o', label='Test Accuracy', markersize=6)
ax.plot(k_range, train_acc_list, 'g--s', label='Train Accuracy', markersize=5, alpha=0.7)
ax.plot(k_range, f1_list, 'r-^', label='F1-Score (weighted)', markersize=6)
ax.axvline(x=best_k, color='purple', linestyle=':', linewidth=2, label=f'Optimum K={best_k}')
ax.set_xlabel('K Değeri', fontsize=12)
ax.set_ylabel('Skor', fontsize=12)
ax.set_title('K Değerine Göre Model Performansı', fontsize=14, fontweight='bold')
ax.set_xticks(list(k_range))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '04_optimum_k.png'), dpi=150)
plt.close()
print("[OK] Optimum K grafiği kaydedildi.")

# ============================================================
# ADIM 7 - Farklı Uzaklık Ölçüleri
# ============================================================
print("\n" + "="*60)
print("ADIM 7: FARKLI UZAKLIK ÖLÇÜLERİ")
print("="*60)

metrics_results = {}
for metric_name, p_val in [('euclidean', 2), ('manhattan', 1), ('minkowski_p3', 3), ('minkowski_p4', 4)]:
    if metric_name.startswith('minkowski'):
        knn_m = KNeighborsClassifier(n_neighbors=best_k, metric='minkowski', p=p_val)
    else:
        knn_m = KNeighborsClassifier(n_neighbors=best_k, metric=metric_name)
    knn_m.fit(X_train, y_train)
    yp_m = knn_m.predict(X_test)
    a = accuracy_score(y_test, yp_m)
    f = f1_score(y_test, yp_m, average='weighted')
    metrics_results[metric_name] = {'accuracy': a, 'f1': f, 'p': p_val}
    print(f"  {metric_name:15s} (p={p_val}) → Accuracy: {a:.4f}, F1: {f:.4f}")

# Uzaklık ölçüleri grafiği
fig, ax = plt.subplots(figsize=(9, 5))
names = list(metrics_results.keys())
accs = [v['accuracy'] for v in metrics_results.values()]
f1s = [v['f1'] for v in metrics_results.values()]
x_pos = np.arange(len(names))
w = 0.35
ax.bar(x_pos - w/2, accs, w, label='Accuracy', color='#3498db', edgecolor='black')
ax.bar(x_pos + w/2, f1s, w, label='F1-Score', color='#e74c3c', edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(['Euclidean\n(p=2)', 'Manhattan\n(p=1)', 'Minkowski\n(p=3)', 'Minkowski\n(p=4)'])
ax.set_ylabel('Skor')
ax.set_title(f'Uzaklık Ölçülerine Göre Performans (K={best_k})', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(max(accs), max(f1s)) + 0.1)
for i, (a, f) in enumerate(zip(accs, f1s)):
    ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
    ax.text(i + w/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, '05_uzaklik_olculeri.png'), dpi=150)
plt.close()
print("[OK] Uzaklık ölçüleri grafiği kaydedildi.")

# En iyi model ile final confusion matrix
print("\n" + "="*60)
print(f"FİNAL: EN İYİ MODEL (K={best_k}, Euclidean)")
print("="*60)
knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_final = knn_final.predict(X_test)
print("\nFinal Classification Report:")
print(classification_report(y_test, y_final, target_names=list(sinif_isimleri.values())))

fig, ax = plt.subplots(figsize=(7, 6))
cm_f = confusion_matrix(y_test, y_final)
disp_f = ConfusionMatrixDisplay(cm_f, display_labels=list(sinif_isimleri.values()))
disp_f.plot(ax=ax, cmap='Greens', colorbar=True)
ax.set_title(f'Final Confusion Matrix (K={best_k})', fontsize=14, fontweight='bold')
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, '06_final_confusion_matrix.png'), dpi=150)
plt.close()

print("\n" + "="*60)
print("TÜM ÇIKTILAR BAŞARIYLA KAYDEDİLDİ!")
print(f"Çıktı klasörü: {OUT}")
print("="*60)
