

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
