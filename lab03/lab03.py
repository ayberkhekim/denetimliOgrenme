import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# GÖREV 1: Veri Ön İşleme (20 Puan)
# ==========================================
# Veri setini yükleyin
file_path = r'C:\Users\ayberk\Desktop\Lab3\Social_Network_Ads.csv'
df = pd.read_csv(file_path)

# Gereksiz sütunları kaldırın (User ID, Gender) [cite: 8]
df = df.drop(['User ID', 'Gender'], axis=1)

# X (Age, EstimatedSalary) ve y (Purchased) olarak ayırın [cite: 5, 8]
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split yapın (%75 train, %25 test) [cite: 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling uygulayın [cite: 8]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "Confusion Matrix": confusion_matrix(y_true, y_pred)
    }


# ==========================================
# GÖREV 2: Logistic Regression Modeli (20 Puan)
# ==========================================
# Modeli eğitin [cite: 10]
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Test verisi üzerinde tahmin yapın [cite: 10]
y_pred_log = log_reg.predict(X_test)

# Metrikleri hesaplayın [cite: 10]
log_metrics = calculate_metrics(y_test, y_pred_log)

# ==========================================
# GÖREV 3: Linear Regression Modeli (25 Puan)
# ==========================================
# Linear Regression modeli kurun
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Tahminleri alın (çıktılar sürekli olacaktır)
y_pred_lin_cont = lin_reg.predict(X_test)

# 0.5 threshold kullanarak sınıfa çevirin
y_pred_lin = (y_pred_lin_cont >= 0.5).astype(int)

# Metrikleri hesaplayın
lin_metrics = calculate_metrics(y_test, y_pred_lin)

# Sonuçları Yazdır
print("--- Logistic Regression Metrics ---")
for k, v in log_metrics.items(): print(f"{k}: {v}")

print("\n--- Linear Regression Metrics (with 0.5 Threshold) ---")
for k, v in lin_metrics.items(): print(f"{k}: {v}")


# ==========================================
# GÖREV 5: Bonus: Görselleştirme (10 Puan)
# ==========================================
def plot_boundary(X_set, y_set, classifier, title):
    X1, X2 = np.meshgrid(np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
                         np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01))

    if isinstance(classifier, LinearRegression):
        Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)
        Z = (Z >= 0.5).astype(int)
    else:
        Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

    plt.contourf(X1, X2, Z.reshape(X1.shape), alpha=0.75, cmap='Paired')
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], label=j)
    plt.title(title)
    plt.xlabel('Age (Scaled)')
    plt.ylabel('Salary (Scaled)')
    plt.legend()
    plt.show()


plot_boundary(X_test, y_test, log_reg, "Logistic Regression Decision Boundary")
plot_boundary(X_test, y_test, lin_reg, "Linear Regression Decision Boundary")