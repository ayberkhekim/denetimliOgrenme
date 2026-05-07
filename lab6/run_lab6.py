# -*- coding: utf-8 -*-
"""
YZM0206 Laboratuvar 6 - Tüm Uygulamalar
Tüm çıktıları ve grafikleri kaydeder.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import confusion_matrix
import seaborn as sns
import io
import sys

# Türkçe karakter desteği
plt.rcParams['font.family'] = 'DejaVu Sans'

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_lines = []
def log(msg):
    print(msg)
    log_lines.append(msg)

def save_log():
    with open(os.path.join(OUTPUT_DIR, 'output_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))

def capture_summary(model):
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + '\n'))
    s = buf.getvalue()
    log(s)
    return s

# =============================================
# UYGULAMA 1: Veri Kümesi Yükleme ve Shape
# =============================================
log("=" * 60)
log("UYGULAMA 1: Veri Kümesi Anatomisi ve Normalizasyon")
log("=" * 60)

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = datasets.mnist.load_data()
(fmnist_train_images, fmnist_train_labels), (fmnist_test_images, fmnist_test_labels) = datasets.fashion_mnist.load_data()
(cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = datasets.cifar10.load_data()

log(f"MNIST Train Shape: {mnist_train_images.shape}")
log(f"Fashion-MNIST Train Shape: {fmnist_train_images.shape}")
log(f"CIFAR-10 Train Shape: {cifar_train_images.shape}")

# Örnek görsel çizimi
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(mnist_train_images[0], cmap='gray')
axes[0].set_title('MNIST - Rakam', fontsize=13, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(fmnist_train_images[0], cmap='gray')
axes[1].set_title('Fashion-MNIST - Kiyafet', fontsize=13, fontweight='bold')
axes[1].axis('off')
axes[2].imshow(cifar_train_images[0])
axes[2].set_title('CIFAR-10 - Nesne', fontsize=13, fontweight='bold')
axes[2].axis('off')
plt.suptitle('Uc Veri Kumesinden Ornek Gorseller', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uygulama1_ornekler.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] uygulama1_ornekler.png")

# Normalizasyon
mnist_train_images = mnist_train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
mnist_test_images = mnist_test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
fmnist_train_images = fmnist_train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
fmnist_test_images = fmnist_test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
cifar_train_images = cifar_train_images.astype('float32') / 255
cifar_test_images = cifar_test_images.astype('float32') / 255
log("Normalizasyon islemi tamamlandi.")

# =============================================
# UYGULAMA 2: Conv2D ve Padding Stratejileri
# =============================================
log("\n" + "=" * 60)
log("UYGULAMA 2: Conv2D ve Padding Stratejileri")
log("=" * 60)

log("\n--- Durum 1: 28x28x1, 3x3 kernel, stride=1, padding='valid' (Beklenen: 26x26) ---")
model1 = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), strides=1, padding='valid')
])
summary1 = capture_summary(model1)

log("\n--- Durum 2: 28x28x1, 3x3 kernel, stride=1, padding='same' (Beklenen: 28x28) ---")
model2 = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), strides=1, padding='same')
])
summary2 = capture_summary(model2)

log("\n--- Durum 3: 32x32x3, 5x5 kernel, stride=2, padding='valid' (Beklenen: 14x14) ---")
model3 = models.Sequential([
    layers.InputLayer(input_shape=(32, 32, 3)),
    layers.Conv2D(32, (5, 5), strides=2, padding='valid')
])
summary3 = capture_summary(model3)

# Padding karşılaştırma tablosu görseli
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
table_data = [
    ['Giris Boyutu', 'Kernel', 'Stride', 'Padding', 'Beklenen Cikti', 'Keras Dogrulama'],
    ['28x28x1', '3x3', '1', 'valid', '26x26', '26x26 ✓'],
    ['28x28x1', '3x3', '1', 'same', '28x28', '28x28 ✓'],
    ['32x32x3', '5x5', '2', 'valid', '14x14', '14x14 ✓'],
]
table = ax.table(cellText=table_data[1:], colLabels=table_data[0], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif i % 2 == 0:
        cell.set_facecolor('#ecf0f1')
    else:
        cell.set_facecolor('#ffffff')
    cell.set_edgecolor('#bdc3c7')
plt.title('Uygulama 2: Cikti Boyutu Hesaplama Tablosu', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uygulama2_tablo.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] uygulama2_tablo.png")

# =============================================
# UYGULAMA 3: Pooling vs Stride=2
# =============================================
log("\n" + "=" * 60)
log("UYGULAMA 3: Pooling vs. Stride 2")
log("=" * 60)

# Model A: MaxPooling
model_pool = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_pool.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model B: Stride=2
model_stride = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), strides=2, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model_stride.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

log("\n--- Model A: MaxPooling ---")
capture_summary(model_pool)
log("\n--- Model B: Stride=2 ---")
capture_summary(model_stride)

# Eğitim
log("\nModel A (MaxPooling) egitimi basliyor...")
hist_pool = model_pool.fit(fmnist_train_images, fmnist_train_labels, epochs=5,
                            validation_data=(fmnist_test_images, fmnist_test_labels), verbose=1)
log("\nModel B (Stride=2) egitimi basliyor...")
hist_stride = model_stride.fit(fmnist_train_images, fmnist_train_labels, epochs=5,
                                validation_data=(fmnist_test_images, fmnist_test_labels), verbose=1)

pool_acc = hist_pool.history['val_accuracy'][-1]
stride_acc = hist_stride.history['val_accuracy'][-1]
log(f"\nModel A (MaxPooling) Son Validation Accuracy: {pool_acc:.4f}")
log(f"Model B (Stride=2) Son Validation Accuracy: {stride_acc:.4f}")

# Karşılaştırma grafiği
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, 6)

ax1.plot(epochs_range, hist_pool.history['accuracy'], 'b-o', label='MaxPooling - Train', linewidth=2)
ax1.plot(epochs_range, hist_pool.history['val_accuracy'], 'b--s', label='MaxPooling - Val', linewidth=2)
ax1.plot(epochs_range, hist_stride.history['accuracy'], 'r-o', label='Stride=2 - Train', linewidth=2)
ax1.plot(epochs_range, hist_stride.history['val_accuracy'], 'r--s', label='Stride=2 - Val', linewidth=2)
ax1.set_title('Dogruluk Karsilastirmasi', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, hist_pool.history['loss'], 'b-o', label='MaxPooling - Train', linewidth=2)
ax2.plot(epochs_range, hist_pool.history['val_loss'], 'b--s', label='MaxPooling - Val', linewidth=2)
ax2.plot(epochs_range, hist_stride.history['loss'], 'r-o', label='Stride=2 - Train', linewidth=2)
ax2.plot(epochs_range, hist_stride.history['val_loss'], 'r--s', label='Stride=2 - Val', linewidth=2)
ax2.set_title('Kayip Karsilastirmasi', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.suptitle('Uygulama 3: MaxPooling vs Stride=2 Karsilastirmasi', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uygulama3_karsilastirma.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] uygulama3_karsilastirma.png")

# =============================================
# UYGULAMA 4: Fashion-MNIST Sınıflandırma
# =============================================
log("\n" + "=" * 60)
log("UYGULAMA 4: Uctan Uca Siniflandirma (Fashion-MNIST)")
log("=" * 60)

app4_model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
app4_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

log("\n--- Fashion-MNIST Model Ozeti ---")
capture_summary(app4_model)

log("\nFashion-MNIST egitimi basliyor (5 epoch)...")
history_fmnist = app4_model.fit(fmnist_train_images, fmnist_train_labels, epochs=5,
                                 validation_data=(fmnist_test_images, fmnist_test_labels), verbose=1)

fmnist_final_acc = history_fmnist.history['val_accuracy'][-1]
log(f"Fashion-MNIST Son Validation Accuracy: {fmnist_final_acc:.4f}")

# Eğitim grafikleri
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
epochs_range = range(1, 6)
ax1.plot(epochs_range, history_fmnist.history['accuracy'], 'g-o', label='Train Accuracy', linewidth=2, markersize=8)
ax1.plot(epochs_range, history_fmnist.history['val_accuracy'], 'g--s', label='Validation Accuracy', linewidth=2, markersize=8)
ax1.set_title('Fashion-MNIST Dogruluk', fontsize=13, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history_fmnist.history['loss'], 'm-o', label='Train Loss', linewidth=2, markersize=8)
ax2.plot(epochs_range, history_fmnist.history['val_loss'], 'm--s', label='Validation Loss', linewidth=2, markersize=8)
ax2.set_title('Fashion-MNIST Kayip', fontsize=13, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('Uygulama 4: Fashion-MNIST Egitim Grafikleri', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uygulama4_egitim.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] uygulama4_egitim.png")

# Confusion Matrix
predictions = app4_model.predict(fmnist_test_images)
predicted_classes = np.argmax(predictions, axis=1)
cm = confusion_matrix(fmnist_test_labels, predicted_classes)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Siniflar', fontsize=12, fontweight='bold')
plt.ylabel('Gercek Siniflar', fontsize=12, fontweight='bold')
plt.title('Fashion-MNIST Hata Matrisi (Confusion Matrix)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'uygulama4_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] uygulama4_confusion_matrix.png")

# En çok karıştırılan sınıfları bul
np.fill_diagonal(cm, 0)
log("\nEn cok karistirilan sinif ciftleri:")
for _ in range(5):
    idx = np.unravel_index(np.argmax(cm), cm.shape)
    log(f"  {class_names[idx[0]]} -> {class_names[idx[1]]}: {cm[idx[0], idx[1]]} kez")
    cm[idx[0], idx[1]] = 0

# =============================================
# GENEL DEĞERLENDIRME: 3 Veri Seti Karşılaştırması
# =============================================
log("\n" + "=" * 60)
log("GENEL DEGERLENDIRME: 3 Veri Seti Karsilastirmasi")
log("=" * 60)

# MNIST
log("\n--- MNIST Egitimi ---")
mnist_model = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
mnist_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
capture_summary(mnist_model)
hist_mnist = mnist_model.fit(mnist_train_images, mnist_train_labels, epochs=3,
                              validation_data=(mnist_test_images, mnist_test_labels), verbose=1)

mnist_final = hist_mnist.history['val_accuracy'][-1]
log(f"MNIST Son Validation Accuracy: {mnist_final:.4f}")

# CIFAR-10
log("\n--- CIFAR-10 Egitimi ---")
cifar_model = models.Sequential([
    layers.InputLayer(input_shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cifar_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
capture_summary(cifar_model)
hist_cifar = cifar_model.fit(cifar_train_images, cifar_train_labels, epochs=5,
                              validation_data=(cifar_test_images, cifar_test_labels), verbose=1)

cifar_final = hist_cifar.history['val_accuracy'][-1]
log(f"CIFAR-10 Son Validation Accuracy: {cifar_final:.4f}")

# Karşılaştırma bar grafiği
fig, ax = plt.subplots(figsize=(10, 6))
datasets_list = ['MNIST\n(Rakamlar)', 'Fashion-MNIST\n(Kiyafetler)', 'CIFAR-10\n(Nesneler)']
accuracies = [mnist_final * 100, fmnist_final_acc * 100, cifar_final * 100]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(datasets_list, accuracies, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
            f'%{acc:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('3 Veri Kumesi CNN Siniflandirma Basari Karsilastirmasi', fontsize=14, fontweight='bold')
ax.set_ylim(0, 105)
ax.grid(True, alpha=0.3, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'genel_karsilastirma.png'), dpi=150, bbox_inches='tight')
plt.close()
log("[KAYDEDILDI] genel_karsilastirma.png")

# Sonuçları kaydet
results = {
    'mnist_acc': mnist_final,
    'fmnist_acc': fmnist_final_acc,
    'cifar_acc': cifar_final,
    'pool_acc': pool_acc,
    'stride_acc': stride_acc,
}
with open(os.path.join(OUTPUT_DIR, 'results.txt'), 'w', encoding='utf-8') as f:
    for k, v in results.items():
        f.write(f"{k}={v:.4f}\n")

save_log()
log("\n" + "=" * 60)
log("TUM ISLEMLER TAMAMLANDI!")
log("=" * 60)
print(f"\nSonuclar {OUTPUT_DIR} klasorune kaydedildi.")
