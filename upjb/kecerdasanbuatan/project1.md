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
