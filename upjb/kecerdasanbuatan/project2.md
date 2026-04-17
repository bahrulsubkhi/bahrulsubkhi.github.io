
# PROJECT 2 — MACHINE LEARNING (KLASIFIKASI, REGRESI, CLUSTERING)

## Deskripsi
Kelompok membangun **tiga model** Machine Learning dalam satu notebook: model Klasifikasi, model Regresi, dan model Clustering. Dataset boleh berbeda untuk setiap model asal relevan.

## Ketentuan Project

### BAGIAN A — KLASIFIKASI

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset klasifikasi
# Contoh: dataset penyakit, spam, churn pelanggan, dll.
df = pd.read_csv('dataset_klasifikasi.csv')

print("Shape dataset:", df.shape)
print("\nInfo dataset:")
print(df.info())
print("\nCek missing value:")
print(df.isnull().sum())

# Encoding label jika kategorikal
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])

# Pisahkan fitur dan target
X = df.drop('target', axis=1)
y = df['target']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# --- Model 1: Decision Tree ---
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("=== DECISION TREE ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

# --- Model 2: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== RANDOM FOREST ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# Perbandingan accuracy
algoritma = ['Decision Tree', 'Random Forest']
akurasi   = [accuracy_score(y_test, y_pred_dt), accuracy_score(y_test, y_pred_rf)]

plt.figure(figsize=(7, 4))
plt.bar(algoritma, akurasi, color=['steelblue', 'seagreen'])
plt.title('Perbandingan Accuracy Algoritma Klasifikasi')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, v in enumerate(akurasi):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.tight_layout()
plt.savefig('perbandingan_klasifikasi.png')
plt.show()

# Feature Importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=df.drop('target', axis=1).columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 4), color='coral')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
```

### BAGIAN B — REGRESI

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset regresi
# Contoh: prediksi harga rumah, prediksi nilai mahasiswa, prediksi curah hujan, dll.
df_reg = pd.read_csv('dataset_regresi.csv')

print("Shape dataset regresi:", df_reg.shape)
print(df_reg.describe())

X_reg = df_reg.drop('target', axis=1)
y_reg = df_reg['target']

scaler_reg = StandardScaler()
X_reg_scaled = scaler_reg.fit_transform(X_reg)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg_scaled, y_reg, test_size=0.2, random_state=42
)

# --- Model 1: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train_r, y_train_r)
y_pred_lr = lr.predict(X_test_r)

print("=== LINEAR REGRESSION ===")
print(f"MAE : {mean_absolute_error(y_test_r, y_pred_lr):.4f}")
print(f"MSE : {mean_squared_error(y_test_r, y_pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_lr)):.4f}")
print(f"R²  : {r2_score(y_test_r, y_pred_lr):.4f}")

# --- Model 2: Random Forest Regressor ---
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
rfr.fit(X_train_r, y_train_r)
y_pred_rfr = rfr.predict(X_test_r)

print("\n=== RANDOM FOREST REGRESSOR ===")
print(f"MAE : {mean_absolute_error(y_test_r, y_pred_rfr):.4f}")
print(f"MSE : {mean_squared_error(y_test_r, y_pred_rfr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_rfr)):.4f}")
print(f"R²  : {r2_score(y_test_r, y_pred_rfr):.4f}")

# Visualisasi Aktual vs Prediksi
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_test_r, y_pred_lr, alpha=0.5, color='steelblue')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.title('Linear Regression: Aktual vs Prediksi')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')

plt.subplot(1, 2, 2)
plt.scatter(y_test_r, y_pred_rfr, alpha=0.5, color='seagreen')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--')
plt.title('Random Forest: Aktual vs Prediksi')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')

plt.tight_layout()
plt.savefig('regresi_aktual_vs_prediksi.png')
plt.show()
```

### BAGIAN C — CLUSTERING

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load dataset clustering
# Contoh: segmentasi pelanggan, pengelompokan mahasiswa, dll.
df_clust = pd.read_csv('dataset_clustering.csv')

X_clust = df_clust.select_dtypes(include=[np.number])
scaler_clust = StandardScaler()
X_clust_scaled = scaler_clust.fit_transform(X_clust)

# Elbow Method untuk menentukan jumlah cluster optimal
inertia = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_clust_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertia, marker='o', color='coral')
plt.title('Elbow Method — Menentukan Jumlah Cluster Optimal')
plt.xlabel('Jumlah Cluster (K)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('elbow_method.png')
plt.show()

# Silhouette Score
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_clust_scaled)
    score = silhouette_score(X_clust_scaled, labels)
    print(f"K={k} | Silhouette Score: {score:.4f}")

# Training dengan K optimal (misalnya K=3)
K_optimal = 3
kmeans = KMeans(n_clusters=K_optimal, random_state=42, n_init=10)
df_clust['cluster'] = kmeans.fit_predict(X_clust_scaled)

print("\nJumlah data per cluster:")
print(df_clust['cluster'].value_counts().sort_index())

# Visualisasi Cluster (PCA 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clust_scaled)

plt.figure(figsize=(8, 6))
for c in range(K_optimal):
    idx = df_clust['cluster'] == c
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Cluster {c}', alpha=0.6)

plt.title(f'Visualisasi Cluster K-Means (K={K_optimal}) — PCA 2D')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.tight_layout()
plt.savefig('visualisasi_cluster.png')
plt.show()

# Profil tiap cluster
print("\nRata-rata fitur per cluster:")
print(df_clust.groupby('cluster').mean())
```

---

## KERTAS KERJA PROJECT 2

### A. Teori
1. Jelaskan perbedaan antara Klasifikasi, Regresi, dan Clustering — kapan masing-masing digunakan?
2. Jelaskan cara kerja Decision Tree dan Random Forest serta perbedaannya
3. Jelaskan cara kerja Linear Regression dan apa arti nilai R², MAE, MSE, RMSE
4. Jelaskan cara kerja K-Means Clustering dan bagaimana Elbow Method & Silhouette Score membantu menentukan K optimal

### B. Dokumentasi Code
1. Screenshot setiap sel Jupyter Notebook beserta outputnya
2. Jelaskan fungsi tiap blok code

### C. Analisis Hasil
**Klasifikasi:**
1. Bandingkan accuracy Decision Tree vs Random Forest — mana lebih baik dan mengapa?
2. Fitur mana yang paling berpengaruh dalam prediksi (dari feature importance)?

**Regresi:**
1. Bandingkan MAE, RMSE, dan R² Linear Regression vs Random Forest Regressor
2. Dari grafik aktual vs prediksi, apakah titik-titik sudah mendekati garis diagonal? Apa artinya?

**Clustering:**
1. Dari Elbow Method, di K berapa terjadi "siku"? Apakah sesuai dengan Silhouette Score tertinggi?
2. Dari profil rata-rata tiap cluster, apa karakteristik masing-masing kelompok?
3. Beri nama/label yang bermakna untuk tiap cluster berdasarkan karakteristiknya

### D. Kesimpulan
Kesimpulan dari ketiga model dan pembagian tugas anggota

---
---
