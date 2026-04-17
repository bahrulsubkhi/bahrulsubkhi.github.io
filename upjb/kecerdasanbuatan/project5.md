

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
