# 📚 Materi Ajar: Struktur Data Non-Linear
## Bab 10 — Disjoint Set & Heap
> **Untuk:** Mahasiswa yang belum memahami materi  
> **Level:** Pemula → Menengah  
> **Bahasa Implementasi:** Python

---

## 🗺️ Daftar Isi

1. [Disjoint Set (Union-Find)](#1-disjoint-set-union-find)
   - [Apa itu Disjoint Set?](#11-apa-itu-disjoint-set)
   - [Analogi Sehari-hari](#12-analogi-sehari-hari)
   - [Representasi Data](#13-representasi-data)
   - [Operasi find()](#14-operasi-find)
   - [Operasi union()](#15-operasi-union)
   - [Optimasi: Path Compression](#16-optimasi-path-compression)
   - [Optimasi: Union by Rank](#17-optimasi-union-by-rank)
   - [Implementasi Lengkap](#18-implementasi-lengkap)
   - [Analisis Kompleksitas](#19-analisis-kompleksitas)
2. [Heap](#2-heap)
   - [Apa itu Heap?](#21-apa-itu-heap)
   - [Analogi Sehari-hari](#22-analogi-sehari-hari)
   - [Jenis-jenis Heap](#23-jenis-jenis-heap)
   - [Struktur Binary Heap](#24-struktur-binary-heap)
   - [Operasi Push](#25-operasi-push-sift-up)
   - [Operasi Pop](#26-operasi-pop-sift-down)
   - [Operasi Top](#27-operasi-top)
   - [Implementasi Lengkap](#28-implementasi-lengkap)
   - [Build Heap Efisien O(n)](#29-build-heap-efisien-on)
   - [Catatan Implementasi](#210-catatan-implementasi)
3. [Latihan Soal](#3-latihan-soal)
4. [Rangkuman Cepat](#4-rangkuman-cepat)

---

# 1. Disjoint Set (Union-Find)

## 1.1 Apa itu Disjoint Set?

**Disjoint Set** adalah struktur data yang mengelola sekumpulan elemen yang terbagi menjadi beberapa **himpunan yang tidak saling beririsan** (disjoint = terpisah).

> 💡 **Kata kunci:** "Disjoint" = tidak ada elemen yang muncul di dua kelompok sekaligus.

Misalkan kita punya 9 elemen: `{0, 1, 2, 3, 4, 5, 6, 7, 8}`.

Mereka bisa dibagi menjadi 3 kelompok:
```
Kelompok A: {0, 1, 2}
Kelompok B: {3, 4, 5}
Kelompok C: {6, 7, 8}
```

Tidak ada satu elemen pun yang masuk ke dua kelompok sekaligus. Itulah Disjoint Set.

Struktur data ini mendukung **dua operasi utama:**

| Operasi | Fungsi |
|---------|--------|
| `find(x)` | Cari kelompok mana yang memuat elemen `x` |
| `union(x, y)` | Gabungkan kelompok yang memuat `x` dan `y` |

---

## 1.2 Analogi Sehari-hari

Bayangkan kamu adalah **koordinator kelas** yang mengatur mahasiswa ke dalam kelompok praktikum.

- Awalnya setiap mahasiswa = kelompok sendiri-sendiri.
- `union(Ali, Budi)` → Ali dan Budi sekarang satu kelompok.
- `union(Budi, Citra)` → Citra masuk kelompok yang sama dengan Ali & Budi.
- `find(Ali)` → Cari siapa **ketua kelompok** (representatif) dari kelompok Ali.
- `find(Citra)` → Hasilnya sama dengan `find(Ali)` → mereka satu kelompok!

> ✅ Jika `find(x) == find(y)`, maka `x` dan `y` berada dalam kelompok yang sama.

---

## 1.3 Representasi Data

Disjoint Set disimpan dalam **array `parent[]`**. Setiap elemen menyimpan indeks dari *parent*-nya. Jika `parent[i] == i`, maka `i` adalah **root** (representatif/ketua) dari kelompoknya.

### Contoh awal (9 elemen, masing-masing kelompok sendiri):

```
Indeks:  0  1  2  3  4  5  6  7  8
parent:  0  1  2  3  4  5  6  7  8
                 ↑
          parent[i] == i → i adalah root
```

Setelah `union(0,1)` dan `union(1,2)` dan `union(3,4)`:

```
Indeks:  0  1  2  3  4  5  6  7  8
parent:  0  0  0  3  3  5  6  7  8
         ↑        ↑        ↑  ↑  ↑
       root      root    roots (masih sendiri)
```

Artinya: elemen 1 dan 2 memiliki parent 0 → mereka satu kelompok dengan root 0.

---

## 1.4 Operasi `find()`

`find(x)` mencari **root** dari kelompok yang mengandung `x`, dengan cara terus naik ke parent hingga menemukan node yang menjadi parent dirinya sendiri.

### Versi naif (tanpa optimasi):

```python
def find(parent, x):
    # Terus naik ke parent sampai menemukan root
    while parent[x] != x:
        x = parent[x]
    return x
```

### Cara kerjanya — contoh:

```
parent = [0, 0, 1, 3, 3, 5, 6, 7, 8]

find(2):
  - parent[2] = 1  → naik ke 1
  - parent[1] = 0  → naik ke 0
  - parent[0] = 0  → ini root! kembalikan 0

Jadi find(2) = 0
```

### ⚠️ Masalah versi naif:

Jika pohon berbentuk rantai panjang, `find()` bisa memakan waktu **O(n)** — sangat lambat!

```
0 ← 1 ← 2 ← 3 ← 4 ← 5   (rantai panjang)
find(5) harus naik 5 langkah!
```

Solusinya? **Path Compression** (lihat bagian 1.6).

---

## 1.5 Operasi `union()`

`union(x, y)` menggabungkan dua kelompok dengan cara: cari root dari masing-masing elemen, lalu jadikan satu root sebagai parent dari root lainnya.

### Versi naif:

```python
def union(parent, x, y):
    rx = find(parent, x)   # root dari kelompok x
    ry = find(parent, y)   # root dari kelompok y
    
    if rx == ry:
        return  # sudah satu kelompok, tidak perlu dilakukan apa-apa
    
    # Jadikan root x sebagai parent dari root y
    parent[ry] = rx
```

### Contoh langkah demi langkah:

```
Awal: parent = [0, 1, 2, 3, 4]
      Kelompok: {0}, {1}, {2}, {3}, {4}

union(0, 1):
  rx = find(0) = 0
  ry = find(1) = 1
  parent[1] = 0  →  parent = [0, 0, 2, 3, 4]
  Kelompok: {0,1}, {2}, {3}, {4}

union(1, 2):
  rx = find(1) = 0   (naik dari 1 → 0)
  ry = find(2) = 2
  parent[2] = 0  →  parent = [0, 0, 0, 3, 4]
  Kelompok: {0,1,2}, {3}, {4}

union(3, 4):
  rx = find(3) = 3
  ry = find(4) = 4
  parent[4] = 3  →  parent = [0, 0, 0, 3, 3]
  Kelompok: {0,1,2}, {3,4}
```

---

## 1.6 Optimasi: Path Compression

**Ide:** Saat kita memanggil `find(x)`, kita sudah tahu root-nya. Kenapa tidak langsung jadikan semua node di jalur itu **langsung menunjuk ke root**?

Ini membuat operasi `find()` berikutnya jauh lebih cepat!

### Sebelum path compression:

```
Pohon:   0
         |
         1
         |
         2
         |
         3
         |
         4

find(4) → naik: 4→3→2→1→0  (4 langkah)
```

### Sesudah path compression (setelah find(4)):

```
Pohon:   0
       / | \ \
      1  2  3  4   ← semua langsung ke root!

find(4) selanjutnya → langsung ke 0  (1 langkah)
```

### Implementasi dengan rekursi:

```python
def find(parent, x):
    if parent[x] != x:
        # Rekursif: cari root, lalu langsung hubungkan x ke root
        parent[x] = find(parent, parent[x])  # ← PATH COMPRESSION
    return parent[x]
```

### Implementasi iteratif:

```python
def find(parent, x):
    # Langkah 1: Temukan root
    root = x
    while parent[root] != root:
        root = parent[root]
    
    # Langkah 2: Kompres jalur — semua node langsung ke root
    while parent[x] != root:
        next_node = parent[x]
        parent[x] = root   # langsung ke root
        x = next_node
    
    return root
```

---

## 1.7 Optimasi: Union by Rank

**Masalah union naif:** Kita selalu jadikan root x sebagai parent root y, tanpa peduli ukuran pohon. Ini bisa bikin pohon tidak seimbang.

**Ide Union by Rank:** Selalu gabungkan **pohon yang lebih pendek** ke bawah **pohon yang lebih tinggi**. Simpan `rank[]` untuk memperkirakan tinggi pohon.

```python
# Inisialisasi
rank = [0] * n   # semua pohon awalnya tinggi 0

def union(parent, rank, x, y):
    rx = find(parent, x)
    ry = find(parent, y)
    
    if rx == ry:
        return   # sudah satu kelompok
    
    # Pohon lebih pendek → di bawah pohon lebih tinggi
    if rank[rx] < rank[ry]:
        rx, ry = ry, rx   # tukar: rx selalu yang lebih tinggi
    
    parent[ry] = rx   # ry masuk di bawah rx
    
    # Jika tinggi sama, tingkatkan rank rx
    if rank[rx] == rank[ry]:
        rank[rx] += 1
```

### Ilustrasi mengapa penting:

```
Tanpa union by rank:         Dengan union by rank:
union({A,B,C}, {D}):         union({A,B,C}, {D}):

    A                              A
   / \                            / \
  B   C                          B   C
  |                              |
  D   ← pohon tinggi 3           D (sama, tidak masalah)

Tapi:
union({D}, {A,B,C}):   ← SALAH arah!

D
|
A        ← pohon jadi tinggi 4!
/ \
B   C

Union by rank mencegah ini → selalu A jadi root, D di bawah.
```

---

## 1.8 Implementasi Lengkap

```python
class DisjointSet:
    """
    Implementasi Disjoint Set dengan:
    - Path Compression (untuk operasi find)
    - Union by Rank (untuk operasi union)
    """
    
    def __init__(self, n):
        """
        Inisialisasi n elemen (0 hingga n-1).
        Setiap elemen awalnya adalah kelompok sendiri.
        """
        self.parent = list(range(n))  # parent[i] = i (self-loop = root)
        self.rank   = [0] * n         # tinggi pohon awal = 0
        self.count  = n               # jumlah kelompok awal = n
    
    def find(self, x):
        """
        Cari root dari kelompok yang mengandung x.
        Dengan path compression: O(α(n))
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]
    
    def union(self, x, y):
        """
        Gabungkan kelompok yang mengandung x dan y.
        Dengan union by rank: O(α(n))
        Kembalikan True jika berhasil digabung, False jika sudah satu kelompok.
        """
        rx = self.find(x)
        ry = self.find(y)
        
        if rx == ry:
            return False   # sudah satu kelompok
        
        # Gabungkan pohon lebih pendek ke pohon lebih tinggi
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        
        self.parent[ry] = rx
        
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        
        self.count -= 1   # satu kelompok berkurang
        return True
    
    def connected(self, x, y):
        """
        Cek apakah x dan y berada dalam kelompok yang sama.
        """
        return self.find(x) == self.find(y)
    
    def group_count(self):
        """
        Kembalikan jumlah kelompok saat ini.
        """
        return self.count


# ── Contoh Penggunaan ────────────────────────────────────────────────────────

ds = DisjointSet(6)   # 6 elemen: 0, 1, 2, 3, 4, 5

print(ds.connected(0, 5))   # False — belum terhubung

ds.union(0, 1)
ds.union(1, 2)
ds.union(3, 4)

print(ds.connected(0, 2))   # True  — satu kelompok
print(ds.connected(0, 3))   # False — kelompok berbeda
print(ds.group_count())     # 3 → {0,1,2}, {3,4}, {5}

ds.union(2, 3)
print(ds.connected(0, 4))   # True  — sekarang satu kelompok besar
print(ds.group_count())     # 2 → {0,1,2,3,4}, {5}
```

---

## 1.9 Analisis Kompleksitas

### Ackermann Inverse — α(n)

Dengan gabungan **Path Compression + Union by Rank**, kompleksitas setiap operasi adalah:

$$O(\alpha(n))$$

Di mana `α(n)` adalah **fungsi invers Ackermann** — tumbuhnya sangat lambat sekali.

| n (jumlah elemen) | nilai α(n) |
|-------------------|-----------|
| 1 – 4 | 0 |
| 5 – 16 | 1 |
| 17 – 65536 | 2 |
| 65537 – 2^65536 | 3 |
| Hampir tak terbatas | 4 |

> 🔑 **Kesimpulan praktis:** Untuk semua ukuran data yang mungkin ada di dunia nyata, `α(n) ≤ 4`. Artinya setiap operasi berjalan dalam waktu **hampir konstan** — lebih cepat dari O(log n) sekalipun!

### Tabel Perbandingan:

| Implementasi | find() | union() |
|---|---|---|
| Naif (tanpa optimasi) | O(n) | O(n) |
| Hanya Path Compression | O(log n) amortized | O(log n) |
| Hanya Union by Rank | O(log n) | O(log n) |
| **Keduanya ✅** | **O(α(n))** | **O(α(n))** |

### Contoh Penggunaan Nyata:

- **Algoritma Kruskal** — membangun Minimum Spanning Tree
- **Deteksi Siklus** pada graf tidak berarah
- **Percolation Problem** — fisika komputasi
- **Connected Components** — komponen terhubung dalam graf
- **Segmentasi Citra** — mengelompokkan piksel yang bertetangga

---
---

# 2. Heap

## 2.1 Apa itu Heap?

**Heap** adalah **pohon biner yang hampir lengkap** *(complete binary tree)* yang memenuhi **properti heap**: nilai setiap node selalu lebih besar (atau lebih kecil) dari semua nilai di subtree-nya.

Dua syarat Heap:
1. **Complete Binary Tree:** Semua level penuh, kecuali level terakhir yang diisi dari kiri.
2. **Heap Property:** Ada dua jenis:
   - **Max-Heap:** `parent ≥ child` (root = elemen terbesar)
   - **Min-Heap:** `parent ≤ child` (root = elemen terkecil)

> ⚠️ **Heap ≠ BST (Binary Search Tree)!**
> Di BST: nilai kiri < parent < nilai kanan (ada urutan kiri-kanan).
> Di Heap: tidak ada aturan kiri-kanan, hanya parent vs child.

---

## 2.2 Analogi Sehari-hari

Bayangkan **kompetisi beregu**. Di Max-Heap:

- **Root** = juara 1 (nilai terbesar).
- Setiap orang **lebih jagoan dari semua anak buahnya**, tapi tidak perlu lebih jagoan dari tetangganya.
- Kamu selalu tahu siapa juaranya (root) — langsung, tanpa mencari!

Atau bayangkan **antrian prioritas rumah sakit**:
- Pasien dengan kondisi terparah (prioritas tertinggi) selalu di depan.
- Saat pasien baru datang, dia masuk di tempat yang sesuai prioritasnya.
- Saat dokter memanggil, pasien paling parah yang dipanggil.

---

## 2.3 Jenis-jenis Heap

### Max-Heap:
```
         90          ← root: nilai terbesar
        /  \
      75    80
     / \   / \
   50  55 60  65

Setiap parent ≥ kedua anaknya ✅
90 ≥ 75, 80 ✅
75 ≥ 50, 55 ✅
80 ≥ 60, 65 ✅
```

### Min-Heap:
```
         10          ← root: nilai terkecil
        /  \
      25    15
     / \   / \
   50  40 30  35

Setiap parent ≤ kedua anaknya ✅
10 ≤ 25, 15 ✅
25 ≤ 50, 40 ✅
15 ≤ 30, 35 ✅
```

---

## 2.4 Struktur Binary Heap

Kehebatan Heap: **tidak butuh pointer**! Disimpan langsung di **array biasa**.

Untuk node di indeks `i`:
- **Parent:** `(i - 1) // 2`
- **Left child:** `2 * i + 1`
- **Right child:** `2 * i + 2`

### Ilustrasi pemetaan pohon → array:

```
Pohon:          90[0]
               /      \
           75[1]       80[2]
           /   \       /   \
        50[3] 55[4] 60[5] 65[6]

Array: [90, 75, 80, 50, 55, 60, 65]
Index:   0   1   2   3   4   5   6
```

### Verifikasi rumus:

```
Node 75 ada di indeks 1:
  - parent(1)      = (1-1)//2 = 0  → 90 ✅
  - left_child(1)  = 2*1+1   = 3  → 50 ✅
  - right_child(1) = 2*1+2   = 4  → 55 ✅

Node 80 ada di indeks 2:
  - parent(2)      = (2-1)//2 = 0  → 90 ✅
  - left_child(2)  = 2*2+1   = 5  → 60 ✅
  - right_child(2) = 2*2+2   = 6  → 65 ✅
```

> 💡 **Mengapa array?** Karena array lebih *cache-friendly*, hemat memori (tidak ada overhead pointer), dan akses indeks langsung tanpa traversal.

---

## 2.5 Operasi Push (Sift Up)

**Push** = menambahkan elemen baru ke dalam heap.

### Langkah-langkah:
1. Tambahkan elemen baru di **akhir array** (posisi paling bawah-kiri pohon).
2. **Sift Up:** Bandingkan dengan parent-nya. Jika melanggar heap property, tukar.
3. Ulangi langkah 2 ke atas sampai heap property terpenuhi atau mencapai root.

### Contoh: Push 95 ke Max-Heap `[90, 75, 80, 50, 55, 60, 65]`

```
Langkah 1 — Tambah di akhir:
[90, 75, 80, 50, 55, 60, 65, 95]
                              ↑ baru

Pohon:          90
               /  \
             75    80
            / \   / \
          50  55 60  65
          /
         95 ← baru (indeks 7)

Langkah 2 — Sift Up:
  95 di indeks 7 → parent = (7-1)//2 = 3 → nilai 50
  95 > 50 → TUKAR!

[90, 75, 80, 95, 55, 60, 65, 50]
                ↑tukar↑

Langkah 3 — Lanjut sift up:
  95 di indeks 3 → parent = (3-1)//2 = 1 → nilai 75
  95 > 75 → TUKAR!

[90, 95, 80, 75, 55, 60, 65, 50]

Langkah 4 — Lanjut sift up:
  95 di indeks 1 → parent = (1-1)//2 = 0 → nilai 90
  95 > 90 → TUKAR!

[95, 90, 80, 75, 55, 60, 65, 50]

Langkah 5 — Sudah di root (indeks 0), selesai!

Pohon akhir:    95  ← root baru
               /  \
             90    80
            / \   / \
          75  55 60  65
          /
         50
```

### Kode `_sift_up`:

```python
def _sift_up(self, i):
    """Naikkan elemen di indeks i ke posisi yang tepat."""
    parent = (i - 1) // 2
    
    while i > 0 and self.heap[i] > self.heap[parent]:  # Max-Heap
        # Tukar dengan parent
        self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
        # Naik ke atas
        i = parent
        parent = (i - 1) // 2
```

**Kompleksitas:** O(log n) — paling jauh naik setinggi pohon.

---

## 2.6 Operasi Pop (Sift Down)

**Pop** = menghapus dan mengembalikan elemen root (maksimum di Max-Heap).

### Mengapa tidak langsung hapus root?
Jika root dihapus begitu saja, pohon menjadi tidak terhubung. Solusinya: **ganti root dengan elemen terakhir**, lalu turunkan elemen itu ke posisi yang tepat.

### Langkah-langkah:
1. Simpan nilai root (yang akan dikembalikan).
2. Pindahkan **elemen terakhir** ke posisi root.
3. Hapus elemen terakhir (ukuran array berkurang 1).
4. **Sift Down:** Bandingkan root baru dengan child-nya. Tukar dengan child terbesar yang lebih besar.
5. Ulangi ke bawah sampai heap property terpenuhi.

### Contoh: Pop dari Max-Heap `[90, 75, 80, 50, 55, 60, 65]`

```
Langkah 1 — Simpan root: nilai = 90

Langkah 2 & 3 — Pindah elemen terakhir ke root:
[65, 75, 80, 50, 55, 60]
 ↑ elemen terakhir (65) gantikan root

Pohon sementara:   65  ← sementara di root (melanggar heap!)
                  /  \
                75    80
               / \   /
             50  55 60

Langkah 4 — Sift Down dari indeks 0 (nilai 65):
  Children: indeks 1 (75) dan indeks 2 (80)
  Child terbesar: 80 (indeks 2)
  65 < 80 → TUKAR!

[80, 75, 65, 50, 55, 60]

Pohon:          80
               /  \
             75    65
            / \   /
          50  55 60

Langkah 5 — Lanjut sift down dari indeks 2 (nilai 65):
  Children: indeks 5 (60) — tidak ada indeks 6
  65 > 60 → TIDAK perlu tukar, selesai!

Hasil akhir: [80, 75, 65, 50, 55, 60]
Nilai yang dikembalikan: 90 ✅
```

### Kode `_sift_down`:

```python
def _sift_down(self, i):
    """Turunkan elemen di indeks i ke posisi yang tepat."""
    n = len(self.heap)
    
    while True:
        largest = i           # asumsi elemen saat ini adalah terbesar
        left    = 2 * i + 1   # indeks left child
        right   = 2 * i + 2   # indeks right child
        
        # Cek apakah left child lebih besar
        if left < n and self.heap[left] > self.heap[largest]:
            largest = left
        
        # Cek apakah right child lebih besar
        if right < n and self.heap[right] > self.heap[largest]:
            largest = right
        
        # Jika elemen saat ini sudah terbesar, selesai
        if largest == i:
            break
        
        # Tukar dengan child terbesar
        self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
        i = largest   # lanjutkan ke bawah
```

**Kompleksitas:** O(log n) — paling jauh turun setinggi pohon.

---

## 2.7 Operasi Top

**Top** = melihat nilai maksimum/minimum tanpa menghapusnya.

```python
def top(self):
    """Kembalikan elemen terbesar (Max-Heap) tanpa menghapus."""
    if not self.heap:
        return None
    return self.heap[0]   # selalu ada di indeks 0!
```

**Kompleksitas:** O(1) — langsung akses indeks 0.

> 💡 Root heap **selalu** menyimpan nilai maksimum (Max-Heap) atau minimum (Min-Heap). Tidak perlu mencari ke mana-mana!

---

## 2.8 Implementasi Lengkap

```python
class MaxHeap:
    """
    Implementasi Max-Heap menggunakan array.
    Elemen terbesar selalu ada di posisi root (indeks 0).
    """
    
    def __init__(self):
        self.heap = []
    
    # ── Akses ──────────────────────────────────────────────────────────────
    
    def top(self):
        """Kembalikan nilai terbesar. O(1)"""
        return self.heap[0] if self.heap else None
    
    def size(self):
        return len(self.heap)
    
    def is_empty(self):
        return len(self.heap) == 0
    
    # ── Operasi Utama ───────────────────────────────────────────────────────
    
    def push(self, value):
        """Tambahkan value ke heap. O(log n)"""
        self.heap.append(value)            # 1. tambah di akhir
        self._sift_up(len(self.heap) - 1)  # 2. naikkan ke posisi tepat
    
    def pop(self):
        """Hapus dan kembalikan nilai terbesar. O(log n)"""
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        root = self.heap[0]                # simpan nilai root
        self.heap[0] = self.heap.pop()     # pindahkan elemen terakhir ke root
        self._sift_down(0)                 # turunkan ke posisi tepat
        return root
    
    # ── Helper: Sift ────────────────────────────────────────────────────────
    
    def _sift_up(self, i):
        """Naikkan elemen indeks i ke posisi yang sesuai. O(log n)"""
        parent = (i - 1) // 2
        while i > 0 and self.heap[i] > self.heap[parent]:
            self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
            i = parent
            parent = (i - 1) // 2
    
    def _sift_down(self, i):
        """Turunkan elemen indeks i ke posisi yang sesuai. O(log n)"""
        n = len(self.heap)
        while True:
            largest = i
            l, r = 2 * i + 1, 2 * i + 2
            
            if l < n and self.heap[l] > self.heap[largest]:
                largest = l
            if r < n and self.heap[r] > self.heap[largest]:
                largest = r
            
            if largest == i:
                break
            
            self.heap[i], self.heap[largest] = self.heap[largest], self.heap[i]
            i = largest
    
    def __repr__(self):
        return f"MaxHeap({self.heap})"


# ── Contoh Penggunaan ────────────────────────────────────────────────────────

h = MaxHeap()

# Push beberapa nilai
for val in [50, 30, 80, 10, 75, 60, 90]:
    h.push(val)
    print(f"push({val:2d}) → top = {h.top()}, heap = {h.heap}")

print()
print("── Pop semua elemen (urutan turun) ──")
while not h.is_empty():
    print(f"pop() = {h.pop()}")
```

**Output:**
```
push(50) → top = 50, heap = [50]
push(30) → top = 50, heap = [50, 30]
push(80) → top = 80, heap = [80, 30, 50]
push(10) → top = 80, heap = [80, 30, 50, 10]
push(75) → top = 80, heap = [80, 75, 50, 10, 30]
push(60) → top = 80, heap = [80, 75, 60, 10, 30, 50]
push(90) → top = 90, heap = [90, 75, 80, 10, 30, 50, 60]

── Pop semua elemen (urutan turun) ──
pop() = 90
pop() = 80
pop() = 75
pop() = 60
pop() = 50
pop() = 30
pop() = 10
```

> 💡 **Heap Sort!** Pop semua elemen dari Max-Heap → hasilnya urutan menurun. Ini adalah algoritma **Heap Sort** dengan kompleksitas O(n log n).

---

## 2.9 Build Heap Efisien O(n)

### Masalah dengan push satu-satu:

Jika kita punya array `[3, 1, 6, 5, 2, 4]` dan ingin membangun heap, push satu-satu membutuhkan **O(n log n)**.

### Algoritma Floyd — O(n):

Kuncinya: **Node daun (leaf)** tidak perlu di-sift-down (sudah memenuhi heap property karena tidak punya anak). Hanya **node internal (bukan daun)** yang perlu di-sift-down.

Node daun pertama ada di indeks `n//2`. Jadi mulai sift-down dari indeks `n//2 - 1` (node internal terakhir) mundur ke root (indeks 0).

```python
def build_heap(arr):
    """
    Bangun Max-Heap dari array sembarang secara efisien.
    Kompleksitas: O(n)
    """
    n = len(arr)
    
    # Mulai dari node internal terakhir, mundur ke root
    # Node daun pertama = indeks n//2
    # Jadi node internal terakhir = indeks n//2 - 1
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, i, n)
    
    return arr


def _sift_down(arr, i, n):
    """Sift down untuk array langsung."""
    while True:
        largest = i
        l, r = 2 * i + 1, 2 * i + 2
        
        if l < n and arr[l] > arr[largest]:
            largest = l
        if r < n and arr[r] > arr[largest]:
            largest = r
        
        if largest == i:
            break
        
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest


# ── Contoh ───────────────────────────────────────────────────────────────────

arr = [3, 1, 6, 5, 2, 4]
print("Sebelum:", arr)

# Identifikasi: n=6, node internal terakhir = 6//2-1 = 2
# Iterasi: i=2, i=1, i=0

build_heap(arr)
print("Sesudah:", arr)
# Output: [6, 5, 4, 1, 2, 3] — valid Max-Heap!
```

### Ilustrasi langkah demi langkah:

```
Array awal: [3, 1, 6, 5, 2, 4]

Pohon:          3[0]
               /    \
            1[1]    6[2]
           / \     /
         5[3] 2[4] 4[5]

Node daun: indeks 3,4,5 (tidak perlu sift-down)
Node internal: indeks 0,1,2

─── Langkah 1: i=2 (nilai 6) ───
  Children: indeks 5 (nilai 4)
  6 > 4 → tidak perlu tukar
  Pohon tidak berubah.

─── Langkah 2: i=1 (nilai 1) ───
  Children: indeks 3 (nilai 5), indeks 4 (nilai 2)
  Child terbesar: 5 (indeks 3)
  1 < 5 → TUKAR!
  Array: [3, 5, 6, 1, 2, 4]

─── Langkah 3: i=0 (nilai 3) ───
  Children: indeks 1 (nilai 5), indeks 2 (nilai 6)
  Child terbesar: 6 (indeks 2)
  3 < 6 → TUKAR!
  Array: [6, 5, 3, 1, 2, 4]
  
  Lanjut sift-down dari indeks 2 (nilai 3):
  Children: indeks 5 (nilai 4)
  3 < 4 → TUKAR!
  Array: [6, 5, 4, 1, 2, 3]

Hasil akhir: [6, 5, 4, 1, 2, 3] ✅ Valid Max-Heap!
```

### Mengapa O(n) bukan O(n log n)?

Bayangkan pohon dengan n node:
- **Setengah node** adalah daun (level terbawah) → sift-down 0 langkah.
- **Seperempat node** di level sebelumnya → sift-down maks 1 langkah.
- **Seperdelapan node** → maks 2 langkah.
- ...dst.

Total langkah = `n/2·0 + n/4·1 + n/8·2 + ...` = **O(n)** (deret geometri konvergen).

---

## 2.10 Catatan Implementasi

### 1. Min-Heap vs Max-Heap

Hanya satu perubahan: **ganti tanda perbandingan**.

```python
# Max-Heap: parent LEBIH BESAR dari child
if self.heap[i] > self.heap[parent]:   # sift_up
if arr[l] > arr[largest]:              # sift_down

# Min-Heap: parent LEBIH KECIL dari child
if self.heap[i] < self.heap[parent]:   # sift_up
if arr[l] < arr[largest]:              # sift_down
```

### 2. Menggunakan heapq Python (Built-in)

Python menyediakan modul `heapq` yang mengimplementasikan **Min-Heap**:

```python
import heapq

# ── Min-Heap ──────────────────────────────────────────────────────────────

h = []
heapq.heappush(h, 30)
heapq.heappush(h, 10)
heapq.heappush(h, 50)

print(h[0])            # 10 — melihat minimum (top)
print(heapq.heappop(h))  # 10 — hapus minimum

# Build heap langsung dari list:
data = [3, 1, 6, 5, 2, 4]
heapq.heapify(data)    # O(n) — in-place
print(data)            # [1, 2, 4, 5, 3, 6] — valid Min-Heap

# ── "Max-Heap" dengan heapq: negate values ────────────────────────────────

h = []
for val in [30, 10, 50, 20]:
    heapq.heappush(h, -val)   # simpan nilai negatif

max_val = -heapq.heappop(h)   # -(-50) = 50
print(max_val)   # 50 ✅
```

### 3. Priority Queue dengan objek

```python
import heapq

# Tuple (priority, data) — Python membandingkan tuple secara leksikografis
tasks = []
heapq.heappush(tasks, (3, "cuci piring"))    # prioritas 3
heapq.heappush(tasks, (1, "kerjakan PR"))    # prioritas 1 (tertinggi)
heapq.heappush(tasks, (2, "makan siang"))    # prioritas 2

while tasks:
    priority, task = heapq.heappop(tasks)
    print(f"[P{priority}] {task}")

# Output:
# [P1] kerjakan PR
# [P2] makan siang
# [P3] cuci piring
```

### 4. Heap Sort

```python
def heap_sort(arr):
    """Heap Sort — O(n log n), in-place."""
    n = len(arr)
    
    # Langkah 1: Build Max-Heap — O(n)
    for i in range(n // 2 - 1, -1, -1):
        _sift_down(arr, i, n)
    
    # Langkah 2: Ekstrak elemen satu per satu — O(n log n)
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]   # pindahkan max ke akhir
        _sift_down(arr, 0, end)               # restore heap untuk sisa elemen
    
    return arr

def _sift_down(arr, i, n):
    while True:
        largest, l, r = i, 2*i+1, 2*i+2
        if l < n and arr[l] > arr[largest]: largest = l
        if r < n and arr[r] > arr[largest]: largest = r
        if largest == i: break
        arr[i], arr[largest] = arr[largest], arr[i]
        i = largest

# Test
data = [64, 34, 25, 12, 22, 11, 90]
heap_sort(data)
print(data)  # [11, 12, 22, 25, 34, 64, 90]
```

### 5. Ringkasan Kompleksitas Heap

| Operasi | Kompleksitas |
|---------|-------------|
| `top()` | **O(1)** |
| `push(x)` | **O(log n)** |
| `pop()` | **O(log n)** |
| `build_heap(arr)` | **O(n)** |
| Heap Sort | **O(n log n)** |

---

# 3. Latihan Soal

## 3.1 Disjoint Set — Latihan

**Soal 1 (Mudah):** Diberikan `DisjointSet(5)`. Gambarkan array `parent[]` setelah operasi berikut:
```
union(0, 1)
union(2, 3)
union(1, 3)
```
Apakah `connected(0, 2)` bernilai True atau False?

---

**Soal 2 (Sedang):** Diberikan 6 node dan edge-edge berikut. Gunakan Disjoint Set untuk mendeteksi apakah terdapat siklus:
```
Edge: (0,1), (1,2), (2,3), (3,4), (4,2)
```
*(Petunjuk: siklus terdeteksi saat union(x,y) dipanggil tetapi find(x) == find(y))*

---

**Soal 3 (Sulit):** Berapa jumlah operasi `union` minimal untuk menghubungkan semua n node menjadi satu komponen? Jelaskan alasannya.

---

## 3.2 Heap — Latihan

**Soal 4 (Mudah):** Apakah array berikut merupakan valid Max-Heap?
```
a) [90, 75, 80, 50, 55, 60, 65]
b) [90, 75, 80, 95, 55, 60, 65]
c) [50, 75, 80, 90, 55, 60, 65]
```

---

**Soal 5 (Sedang):** Mulai dari Max-Heap kosong, lakukan push berturut-turut: `[15, 10, 20, 8, 25]`. Gambarkan isi array heap setelah setiap operasi push!

---

**Soal 6 (Sedang):** Diberikan Max-Heap `[90, 75, 80, 50, 55, 60, 65]`. Lakukan `pop()` dua kali. Gambarkan isi array setelah masing-masing pop!

---

**Soal 7 (Sulit):** Tunjukkan langkah demi langkah `build_heap` pada array `[5, 3, 8, 1, 9, 2, 7]`!

---

## 3.3 Kunci Jawaban

<details>
<summary>Klik untuk melihat jawaban Soal 1</summary>

```
Awal:    parent = [0, 1, 2, 3, 4]

union(0,1): rx=0, ry=1 → parent[1]=0
            parent = [0, 0, 2, 3, 4]

union(2,3): rx=2, ry=3 → parent[3]=2
            parent = [0, 0, 2, 2, 4]

union(1,3): rx=find(1)=0, ry=find(3)=2 → parent[2]=0
            parent = [0, 0, 0, 2, 4]

connected(0,2): find(0)=0, find(2)=0 → True ✅
```
</details>

<details>
<summary>Klik untuk melihat jawaban Soal 4</summary>

```
a) [90, 75, 80, 50, 55, 60, 65] → ✅ VALID MAX-HEAP
   90≥75,80; 75≥50,55; 80≥60,65

b) [90, 75, 80, 95, 55, 60, 65] → ❌ BUKAN VALID MAX-HEAP
   75 < 95 (parent lebih kecil dari child, melanggar Max-Heap!)

c) [50, 75, 80, 90, 55, 60, 65] → ❌ BUKAN VALID MAX-HEAP
   50 < 75 dan 50 < 80 (root bukan yang terbesar!)
```
</details>

---

# 4. Rangkuman Cepat

## Disjoint Set (Union-Find)

| Aspek | Keterangan |
|-------|-----------|
| **Fungsi** | Mengelola himpunan saling lepas |
| **Operasi** | `find(x)` — cari root; `union(x,y)` — gabung kelompok |
| **Struktur** | Array `parent[]` dan `rank[]` |
| **Optimasi 1** | Path Compression — pohon lebih flat |
| **Optimasi 2** | Union by Rank — pohon lebih seimbang |
| **Kompleksitas** | O(α(n)) per operasi ≈ O(1) praktis |
| **Use Case** | MST Kruskal, deteksi siklus, connected components |

## Heap

| Aspek | Keterangan |
|-------|-----------|
| **Fungsi** | Priority Queue — akses max/min selalu O(1) |
| **Jenis** | Max-Heap (`parent≥child`) / Min-Heap (`parent≤child`) |
| **Struktur** | Array dengan rumus indeks parent/child |
| **Push** | Tambah di akhir + Sift Up → O(log n) |
| **Pop** | Pindah elemen terakhir ke root + Sift Down → O(log n) |
| **Top** | Langsung `heap[0]` → O(1) |
| **Build Heap** | Floyd Algorithm → O(n) |
| **Use Case** | Priority Queue, Heap Sort, Dijkstra, Prim's MST |

---

## 🔑 Poin-poin Kritis yang Sering Salah

1. **Disjoint Set:** Jangan lupa path compression di `find()` — tanpa ini, performa jauh lebih buruk.
2. **Union by Rank ≠ Union by Size** — rank bukan jumlah elemen, tapi perkiraan tinggi pohon.
3. **Heap ≠ BST** — di heap tidak ada urutan kiri-kanan.
4. **Array Heap dimulai indeks 0** — perhatikan rumus `(i-1)//2` vs `i//2` (1-based).
5. **Pop heap:** Elemen terakhir dipindah ke root *dulu*, baru sift-down. Jangan langsung hapus root!
6. **heapq Python adalah Min-Heap** — untuk Max-Heap, gunakan nilai negatif.
7. **Build Heap O(n)** dimulai dari `n//2 - 1`, bukan dari 0 atau n-1.

---

*Materi ini merupakan bagian dari Bab 10 — Struktur Data Non-Linear.*  
*Pemrograman Lanjut · Struktur Data & Algoritma*
