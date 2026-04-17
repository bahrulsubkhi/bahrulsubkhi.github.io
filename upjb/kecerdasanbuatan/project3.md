
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
