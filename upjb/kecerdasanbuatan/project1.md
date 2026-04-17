# KUMPULAN PROJECT — KECERDASAN BUATAN
**Program Studi:** Teknik Informatika / Sistem Informasi
**Semester:** 5–6
**Bahasa:** Python | Format: Jupyter Notebook (.ipynb)
**Sifat:** Kelompok (2–3 orang)

---

# PROJECT 1 — SENTIMENT ANALYSIS (NLP)

## Deskripsi
Kelompok membangun model Sentiment Analysis menggunakan Python untuk mengklasifikasikan teks menjadi sentimen **positif**, **negatif**, atau **netral**. Dataset bebas dipilih asal berupa teks ulasan, komentar, atau tweet berbahasa Indonesia atau Inggris.

## Ketentuan Project

### 1. Persiapan Dataset
- Gunakan dataset minimal **500 baris** teks berlabel sentimen
- Contoh sumber: Kaggle, Twitter API, Google Play Review, Tokopedia Review
- Tampilkan distribusi label sentimen menggunakan grafik

### 2. Preprocessing Teks
Lakukan semua tahap preprocessing berikut secara berurutan:

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
df = pd.read_csv('dataset_sentimen.csv')
print("Jumlah data:", len(df))
print("Distribusi label:\n", df['label'].value_counts())

# Fungsi preprocessing
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocessing(teks):
    # 1. Lowercase
    teks = teks.lower()
    
    # 2. Hapus URL
    teks = re.sub(r'http\S+|www\S+', '', teks)
    
    # 3. Hapus mention dan hashtag
    teks = re.sub(r'@\w+|#\w+', '', teks)
    
    # 4. Hapus karakter selain huruf dan spasi
    teks = re.sub(r'[^a-zA-Z\s]', '', teks)
    
    # 5. Tokenisasi
    tokens = word_tokenize(teks)
    
    # 6. Hapus stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # 7. Stemming
    tokens = [stemmer.stem(t) for t in tokens]
    
    return ' '.join(tokens)

df['teks_bersih'] = df['teks'].apply(preprocessing)
print("\nContoh hasil preprocessing:")
print(df[['teks', 'teks_bersih']].head())
```

### 3. Feature Extraction (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Split data
X = df['teks_bersih']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data training: {len(X_train)} baris")
print(f"Data testing : {len(X_test)} baris")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"\nUkuran matriks TF-IDF training: {X_train_tfidf.shape}")
```

### 4. Training Model (Naive Bayes)

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Training
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

# Prediksi
y_pred = model_nb.predict(X_test_tfidf)

# Evaluasi
print("=== HASIL EVALUASI NAIVE BAYES ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model_nb.classes_,
            yticklabels=model_nb.classes_)
plt.title('Confusion Matrix — Naive Bayes')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix_nb.png')
plt.show()
```

### 5. Uji Prediksi Teks Baru

```python
def prediksi_sentimen(teks_baru):
    teks_bersih = preprocessing(teks_baru)
    teks_tfidf  = tfidf.transform([teks_bersih])
    hasil       = model_nb.predict(teks_tfidf)[0]
    probabilitas = model_nb.predict_proba(teks_tfidf)[0]
    
    print(f"Teks asli  : {teks_baru}")
    print(f"Teks bersih: {teks_bersih}")
    print(f"Sentimen   : {hasil}")
    print(f"Probabilitas: {dict(zip(model_nb.classes_, probabilitas))}")

# Contoh uji
prediksi_sentimen("Produk ini sangat bagus dan pengirimannya cepat!")
prediksi_sentimen("Kecewa banget, barangnya rusak dan pelayanan buruk.")
prediksi_sentimen("Lumayan lah, sesuai ekspektasi.")
```

---

## KERTAS KERJA PROJECT 1

### A. Teori (jelaskan dengan kata-kata sendiri)
1. Apa itu Sentiment Analysis dan apa kegunaannya di dunia nyata?
2. Jelaskan setiap tahap preprocessing: mengapa lowercase, hapus stopwords, dan stemming diperlukan?
3. Jelaskan cara kerja TF-IDF dalam merepresentasikan teks menjadi angka
4. Jelaskan cara kerja algoritma Naive Bayes untuk klasifikasi teks

### B. Dokumentasi Code
1. Screenshot setiap sel Jupyter Notebook yang berjalan beserta outputnya
2. Jelaskan fungsi tiap blok code dengan komentar singkat

### C. Analisis Hasil
1. Berapa accuracy model yang dihasilkan? Apakah sudah memuaskan?
2. Dari confusion matrix, kelas sentimen mana yang paling sering salah diprediksi? Mengapa?
3. Dari classification report, bandingkan nilai precision, recall, dan F1-score tiap kelas
4. Uji minimal 5 teks baru (berbeda topik), tampilkan hasilnya dan analisis apakah prediksi sudah benar
5. Apa kelemahan model ini dan bagaimana cara meningkatkan performanya?

### D. Kesimpulan
Rangkum temuan kelompok dan pembagian tugas anggota

---
---

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

# PROJECT 3 — NEURAL NETWORK / DEEP LEARNING

## Deskripsi
Kelompok membangun model Neural Network menggunakan Keras/TensorFlow untuk menyelesaikan masalah klasifikasi. Dataset bebas asal berupa data tabular atau gambar sederhana.

## Ketentuan Project

### 1. Persiapan Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)

# Load dataset
df = pd.read_csv('dataset_nn.csv')

print("Shape:", df.shape)
print(df.head())
print("\nDistribusi target:")
print(df['target'].value_counts())

# Preprocessing
X = df.drop('target', axis=1)
y = df['target']

le = LabelEncoder()
y = le.fit_transform(y)
jumlah_kelas = len(le.classes_)
print(f"\nJumlah kelas: {jumlah_kelas} — {le.classes_}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData training : {X_train.shape}")
print(f"Data testing  : {X_test.shape}")
```

### 2. Membangun Arsitektur Neural Network

```python
# Arsitektur model
model = keras.Sequential([
    # Input layer
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden layer 1
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Hidden layer 2
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),

    # Output layer
    layers.Dense(jumlah_kelas, activation='softmax')
])

model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Training Model

```python
# Callback untuk early stopping
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

# Training
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print(f"\nTraining selesai di epoch ke-{len(history.history['loss'])}")
```

### 4. Visualisasi Training

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot Loss
ax1.plot(history.history['loss'], label='Training Loss', color='steelblue')
ax1.plot(history.history['val_loss'], label='Validation Loss', color='coral')
ax1.set_title('Training vs Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot Accuracy
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='steelblue')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='coral')
ax2.set_title('Training vs Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
```

### 5. Evaluasi Model

```python
# Evaluasi pada data test
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss    : {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Prediksi
y_pred_prob = model.predict(X_test)
y_pred      = np.argmax(y_pred_prob, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix — Neural Network')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix_nn.png')
plt.show()
```

### 6. Eksperimen Arsitektur

```python
# Bandingkan dengan arsitektur lebih sederhana
model_simple = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(jumlah_kelas, activation='softmax')
])

model_simple.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

history_simple = model_simple.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

loss_s, acc_s = model_simple.evaluate(X_test, y_test, verbose=0)
print(f"Model Sederhana — Test Accuracy: {acc_s:.4f}")
print(f"Model Kompleks  — Test Accuracy: {acc:.4f}")
```

---

## KERTAS KERJA PROJECT 3

### A. Teori
1. Jelaskan apa itu Neural Network: neuron, layer (input, hidden, output), activation function
2. Apa fungsi Dropout dan BatchNormalization dalam arsitektur di atas?
3. Jelaskan cara kerja optimizer Adam dan loss function sparse_categorical_crossentropy
4. Apa itu Early Stopping dan mengapa digunakan?

### B. Dokumentasi Code
1. Screenshot setiap sel Jupyter Notebook beserta outputnya
2. Gambarkan arsitektur model (layer, jumlah neuron, activation) secara manual atau dengan tool

### C. Analisis Hasil
1. Dari grafik loss dan accuracy, apakah model mengalami overfitting atau underfitting? Jelaskan tanda-tandanya
2. Di epoch ke berapa model berhenti training? Apakah Early Stopping bekerja dengan baik?
3. Bandingkan hasil model kompleks vs model sederhana — mana yang lebih baik?
4. Dari confusion matrix, kelas mana yang paling sering salah? Mengapa?
5. Apa dampak perubahan jumlah neuron, dropout rate, atau epoch terhadap hasil?

### D. Kesimpulan
Kesimpulan dan pembagian tugas anggota

---
---

# PROJECT 4 — COMPUTER VISION

## Deskripsi
Kelompok membangun model klasifikasi gambar menggunakan CNN (Convolutional Neural Network) dengan Keras. Dataset berupa kumpulan gambar dengan minimal 2 kelas.

## Ketentuan Project

### 1. Persiapan Dataset Gambar

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Struktur folder dataset:
# dataset/
#   train/
#     kelas_A/  (gambar...)
#     kelas_B/  (gambar...)
#   test/
#     kelas_A/
#     kelas_B/

IMG_SIZE   = (128, 128)
BATCH_SIZE = 32

# Data Augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load data training
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Load data validasi
val_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Load data test
test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Kelas:", train_generator.class_indices)
print(f"Jumlah gambar training   : {train_generator.samples}")
print(f"Jumlah gambar validasi   : {val_generator.samples}")
print(f"Jumlah gambar test       : {test_generator.samples}")

# Tampilkan contoh gambar
gambar, label = next(train_generator)
plt.figure(figsize=(12, 4))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(gambar[i])
    plt.title(list(train_generator.class_indices.keys())[np.argmax(label[i])])
    plt.axis('off')
plt.suptitle('Contoh Gambar Training')
plt.tight_layout()
plt.savefig('contoh_gambar.png')
plt.show()
```

### 2. Membangun Arsitektur CNN

```python
jumlah_kelas = len(train_generator.class_indices)

model_cnn = keras.Sequential([
    # Block Konvolusi 1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block Konvolusi 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block Konvolusi 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Flatten dan Fully Connected
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    # Output
    layers.Dense(jumlah_kelas, activation='softmax')
])

model_cnn.summary()

model_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 3. Training CNN

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5
)

history_cnn = model_cnn.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[early_stop, reduce_lr]
)
```

### 4. Visualisasi Training dan Evaluasi

```python
# Plot history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history_cnn.history['loss'],     label='Training Loss',     color='steelblue')
ax1.plot(history_cnn.history['val_loss'], label='Validation Loss',   color='coral')
ax1.set_title('Loss per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history_cnn.history['accuracy'],     label='Training Accuracy',   color='steelblue')
ax2.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy', color='coral')
ax2.set_title('Accuracy per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history_cnn.png')
plt.show()

# Evaluasi pada test set
loss_test, acc_test = model_cnn.evaluate(test_generator, verbose=0)
print(f"Test Loss    : {loss_test:.4f}")
print(f"Test Accuracy: {acc_test:.4f}")

# Confusion Matrix
y_true = test_generator.classes
y_pred = np.argmax(model_cnn.predict(test_generator), axis=1)
nama_kelas = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=nama_kelas))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=nama_kelas, yticklabels=nama_kelas)
plt.title('Confusion Matrix — CNN')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png')
plt.show()
```

### 5. Prediksi Gambar Baru

```python
from tensorflow.keras.preprocessing import image

def prediksi_gambar(path_gambar):
    img = image.load_img(path_gambar, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediksi = model_cnn.predict(img_array)[0]
    kelas_prediksi = nama_kelas[np.argmax(prediksi)]
    confidence = np.max(prediksi) * 100

    plt.imshow(img)
    plt.title(f"Prediksi: {kelas_prediksi} ({confidence:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"Kelas prediksi : {kelas_prediksi}")
    print(f"Confidence     : {confidence:.2f}%")
    for i, k in enumerate(nama_kelas):
        print(f"  {k}: {prediksi[i]*100:.2f}%")

# Uji dengan gambar baru
prediksi_gambar('gambar_uji_1.jpg')
prediksi_gambar('gambar_uji_2.jpg')
```

---

## KERTAS KERJA PROJECT 4

### A. Teori
1. Jelaskan cara kerja CNN: apa fungsi layer Conv2D, MaxPooling, Flatten, dan Dense?
2. Apa itu Data Augmentation dan mengapa diperlukan pada dataset gambar?
3. Jelaskan perbedaan CNN dengan Neural Network biasa (dari Project 3)
4. Apa itu transfer learning? (jelaskan konsep, meskipun tidak diimplementasikan)

### B. Dokumentasi Code
1. Screenshot setiap sel Jupyter Notebook beserta outputnya
2. Gambarkan arsitektur CNN secara diagram (Conv → Pool → Conv → Pool → Flatten → Dense)

### C. Analisis Hasil
1. Dari grafik training, apakah model mengalami overfitting? Apa tandanya?
2. Apakah Data Augmentation membantu mengurangi overfitting? Jelaskan dari grafik
3. Dari confusion matrix, kelas mana yang paling sering salah diprediksi?
4. Uji minimal 5 gambar baru, tampilkan hasil dan confidence-nya — apakah model sudah andal?
5. Apa tantangan terbesar dalam project Computer Vision ini?

### D. Kesimpulan
Kesimpulan dan pembagian tugas anggota

---
---

# PROJECT 5 — BUILD MODEL & IMPLEMENTASI (PROJECT AKHIR)

## Deskripsi
Project akhir semester. Kelompok memilih **satu permasalahan nyata**, membangun model AI secara lengkap mulai dari pengumpulan data hingga implementasi dalam bentuk aplikasi sederhana berbasis web (Streamlit) atau antarmuka Python.

## Ketentuan Project

### 1. Pemilihan Tema dan Dataset
- Pilih satu permasalahan nyata yang relevan (kesehatan, pertanian, pendidikan, bisnis, dll.)
- Dataset minimal 1.000 baris dari sumber terpercaya (Kaggle, UCI, data pemerintah, dll.)
- Dokumentasikan sumber dataset dan alasan pemilihan tema

### 2. Pipeline Lengkap

```python
# ============================================================
# TAHAP 1: EKSPLORASI DATA (EDA)
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_final.csv')

print("=== INFO DATASET ===")
print(f"Jumlah baris  : {df.shape[0]}")
print(f"Jumlah kolom  : {df.shape[1]}")
print(f"\nTipe data:\n{df.dtypes}")
print(f"\nMissing value:\n{df.isnull().sum()}")
print(f"\nStatistik deskriptif:\n{df.describe()}")

# Distribusi target
plt.figure(figsize=(6, 4))
df['target'].value_counts().plot(kind='bar', color='steelblue')
plt.title('Distribusi Target')
plt.tight_layout()
plt.savefig('distribusi_target.png')
plt.show()

# Heatmap korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap Korelasi Fitur')
plt.tight_layout()
plt.savefig('heatmap_korelasi.png')
plt.show()
```

```python
# ============================================================
# TAHAP 2: PREPROCESSING & FEATURE ENGINEERING
# ============================================================
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Tangani missing value
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encoding kolom kategorikal
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    if col != 'target':
        df[col] = le.fit_transform(df[col])

# Pisahkan fitur dan target
X = df.drop('target', axis=1)
y = le.fit_transform(df['target'])

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training: {X_train.shape} | Testing: {X_test.shape}")
```

```python
# ============================================================
# TAHAP 3: PERBANDINGAN BEBERAPA MODEL
# ============================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time

model_list = {
    'Decision Tree'       : DecisionTreeClassifier(random_state=42),
    'Random Forest'       : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'   : GradientBoostingClassifier(random_state=42),
    'SVM'                 : SVC(random_state=42)
}

hasil_model = []

for nama_model, model in model_list.items():
    start = time.time()
    model.fit(X_train, y_train)
    waktu = time.time() - start

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted')

    hasil_model.append({
        'Model'   : nama_model,
        'Accuracy': acc,
        'F1-Score': f1,
        'Waktu (s)': round(waktu, 4)
    })

    print(f"{nama_model:25s} | Accuracy: {acc:.4f} | F1: {f1:.4f} | Waktu: {waktu:.4f}s")

df_hasil = pd.DataFrame(hasil_model)
print("\n=== TABEL PERBANDINGAN MODEL ===")
print(df_hasil.to_string(index=False))

# Visualisasi perbandingan
df_hasil.set_index('Model')[['Accuracy', 'F1-Score']].plot(
    kind='bar', figsize=(10, 5), color=['steelblue', 'coral']
)
plt.title('Perbandingan Accuracy & F1-Score Antar Model')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('perbandingan_model.png')
plt.show()
```

```python
# ============================================================
# TAHAP 4: TUNING MODEL TERBAIK
# ============================================================
from sklearn.model_selection import GridSearchCV

# Contoh tuning Random Forest
param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Parameter terbaik: {grid_search.best_params_}")
print(f"Accuracy terbaik (CV): {grid_search.best_score_:.4f}")

model_terbaik = grid_search.best_estimator_
y_pred_tuned  = model_terbaik.predict(X_test)
print(f"Accuracy setelah tuning (test): {accuracy_score(y_test, y_pred_tuned):.4f}")
```

```python
# ============================================================
# TAHAP 5: SIMPAN MODEL
# ============================================================
import joblib

joblib.dump(model_terbaik, 'model_final.pkl')
joblib.dump(scaler,        'scaler_final.pkl')
joblib.dump(le,            'encoder_final.pkl')

print("Model berhasil disimpan!")

# Verifikasi load model
model_loaded  = joblib.load('model_final.pkl')
scaler_loaded = joblib.load('scaler_final.pkl')
print("Model berhasil di-load dan siap digunakan!")
```

### 3. Implementasi Aplikasi (Streamlit)

```python
# Simpan file ini sebagai: app.py
# Jalankan dengan: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model  = joblib.load('model_final.pkl')
scaler = joblib.load('scaler_final.pkl')
le     = joblib.load('encoder_final.pkl')

st.title("🤖 Aplikasi Prediksi — [Nama Tema Kelompok]")
st.write("Aplikasi ini memprediksi [deskripsi singkat] menggunakan model Machine Learning.")

st.sidebar.header("Masukkan Data")

# Ganti fitur di bawah sesuai dataset kelompok
fitur1 = st.sidebar.slider("Fitur 1", min_value=0, max_value=100, value=50)
fitur2 = st.sidebar.slider("Fitur 2", min_value=0, max_value=100, value=50)
fitur3 = st.sidebar.number_input("Fitur 3", min_value=0.0, max_value=100.0, value=50.0)

if st.sidebar.button("🔍 Prediksi"):
    input_data = np.array([[fitur1, fitur2, fitur3]])
    input_scaled = scaler.transform(input_data)
    prediksi = model.predict(input_scaled)[0]
    probabilitas = model.predict_proba(input_scaled)[0]

    hasil = le.inverse_transform([prediksi])[0]

    st.success(f"Hasil Prediksi: **{hasil}**")

    st.write("### Probabilitas Tiap Kelas:")
    df_prob = pd.DataFrame({
        'Kelas'      : le.classes_,
        'Probabilitas': probabilitas
    })
    st.bar_chart(df_prob.set_index('Kelas'))

st.write("---")
st.write("Dikembangkan oleh: [Nama Kelompok] | Mata Kuliah Kecerdasan Buatan")
```

---

## KERTAS KERJA PROJECT 5

### A. Teori
1. Jelaskan permasalahan nyata yang dipilih dan mengapa AI relevan untuk menyelesaikannya
2. Jelaskan pipeline Machine Learning secara lengkap: dari data mentah hingga model siap pakai
3. Jelaskan teknik hyperparameter tuning yang digunakan (GridSearchCV) dan cara kerjanya
4. Jelaskan apa itu deployment model dan mengapa Streamlit dipilih sebagai platform implementasi

### B. Dokumentasi Code
1. Screenshot setiap tahap pipeline (EDA, preprocessing, training, tuning, implementasi)
2. Screenshot tampilan aplikasi Streamlit saat berjalan
3. Jelaskan fungsi tiap blok code

### C. Analisis Hasil
1. Dari EDA: fitur mana yang paling berkorelasi dengan target? Apakah ada outlier atau missing value yang signifikan?
2. Dari perbandingan model: algoritma mana yang terbaik untuk dataset ini? Mengapa?
3. Apakah tuning (GridSearchCV) meningkatkan accuracy? Berapa peningkatannya?
4. Demonstrasikan aplikasi Streamlit dengan minimal 5 skenario input berbeda — apakah hasil prediksi masuk akal?
5. Apa keterbatasan model dan aplikasi yang dibangun?

### D. Kesimpulan
Kesimpulan keseluruhan project akhir, refleksi pembelajaran, dan pembagian tugas anggota

---

# RUBRIK PENILAIAN SEMUA PROJECT

| Komponen | Bobot |
|---|---|
| Kebenaran & kelengkapan implementasi code | 40% |
| Visualisasi dan output yang dihasilkan | 20% |
| Kertas Kerja — Teori | 15% |
| Kertas Kerja — Analisis Hasil | 20% |
| Kertas Kerja — Dokumentasi Code | 5% |
| **Total** | **100%** |

---

# KETENTUAN UMUM SEMUA PROJECT

- Format pengumpulan: file `.zip` berisi **Jupyter Notebook (.ipynb)** + **Kertas Kerja (PDF/fisik)**
- Notebook harus sudah dijalankan (semua sel ada outputnya)
- Setiap kelompok wajib mencantumkan **pembagian tugas** anggota di kertas kerja
- Dataset yang digunakan wajib disertakan dalam file zip
- Dilarang menggunakan dataset dan code yang sama antar kelompok
