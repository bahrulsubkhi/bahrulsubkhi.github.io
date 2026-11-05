# Flappy Bird — Godot 4

**Proyek pembelajaran mata kuliah Pengantar Game · S1 Informatika UPJB · Semester 3**

Proyek ini merupakan contoh game 2D Flappy Bird yang dibangun menggunakan Godot 4 dan GDScript. Versi ini sudah memiliki alur permainan yang lengkap, visual procedural, sistem skor, kondisi menang/kalah, pause, restart, dan penyimpanan best score.

## 1. Fitur Permainan

- **Main Menu** dengan petunjuk kontrol.
- **Gameplay** dengan karakter, gravitasi, flap, pipa, collision, dan HUD.
- **Score System**: pemain memperoleh 1 poin saat berhasil melewati satu pasangan pipa.
- **Win Condition**: pemain menang ketika mencapai 10 poin.
- **Lose Condition**: pemain kalah saat menabrak pipa atau keluar dari batas arena.
- **Pause / Resume** melalui tombol JEDA atau tombol ESC.
- **Result Screen** yang membedakan LEVEL CLEAR dan GAME OVER.
- **Restart** dan kembali ke Main Menu.
- **Best Score** disimpan secara lokal melalui `user://savegame.cfg`.
- **Difficulty Scaling**: pipa bergerak semakin cepat dan celah berkurang secara bertahap.

## 2. Menjalankan Proyek

1. Gunakan Godot Engine 4.7.x.
2. Buka Godot Project Manager.
3. Klik **Import**.
4. Pilih file `project.godot` pada folder proyek ini.
5. Buka proyek, kemudian tekan **F6** untuk menjalankan scene aktif atau **F5** untuk menjalankan dari Main Menu.

### Kontrol

| Aksi | Kontrol |
|---|---|
| Mengepak / mulai | SPACE, W, PANAH ATAS, klik kiri |
| Jeda | Tombol JEDA atau ESC |
| Melanjutkan | Tombol LANJUTKAN atau ESC |
| Mengulang | Tombol ULANGI PERMAINAN |
| Kembali ke menu | Tombol MENU UTAMA |

## 3. Struktur Proyek

```text
FlappyBird-Godot4/
├── project.godot
├── README.md
├── scenes/
│   ├── menu.tscn
│   ├── main.tscn
│   ├── result.tscn
│   ├── bird.tscn
│   └── pipe_pair.tscn
├── scripts/
│   ├── game_data.gd
│   ├── menu.gd
│   ├── main.gd
│   ├── result.gd
│   ├── bird.gd
│   └── pipe_pair.gd
└── tests/
    └── smoke_test.gd
```

### Tanggung Jawab Scene dan Script

| File | Tanggung jawab |
|---|---|
| `menu.tscn` + `menu.gd` | Menu pembuka, informasi kontrol, tombol mulai |
| `main.tscn` + `main.gd` | Gameplay, game loop, FSM, spawn pipa, HUD, pause |
| `result.tscn` + `result.gd` | Tampilan hasil menang/kalah, restart, kembali ke menu |
| `bird.tscn` + `bird.gd` | Karakter, gravitasi, flap, rotasi, collision |
| `pipe_pair.tscn` + `pipe_pair.gd` | Pipa atas-bawah, collision solid, trigger skor |
| `game_data.gd` | Autoload untuk skor, target, hasil run, dan best score |
| `tests/smoke_test.gd` | Uji otomatis geometri pipa, trigger skor, dan transisi scene |

## 4. Alur Scene dan Game State

### Alur perpindahan scene

```text
MAIN MENU
    |
    | Mulai Bermain
    v
GAMEPLAY
    |
    +---- Skor mencapai 10 ----------> RESULT: LEVEL CLEAR
    |
    +---- Menabrak / keluar arena ---> RESULT: GAME OVER

RESULT
    |
    +---- Main Lagi -----------------> GAMEPLAY
    |
    +---- Kembali ke Menu ----------> MAIN MENU
```

### Finite State Machine Gameplay

- **READY**: karakter menunggu input pertama.
- **PLAYING**: karakter dan pipa bergerak; skor dapat bertambah.
- **PAUSED**: karakter dan pipa berhenti sementara.
- **FINISHED**: permainan berakhir dan hasil disiapkan.

State digunakan agar input, pergerakan, skor, dan UI hanya berjalan pada kondisi yang tepat.

## 5. Konsep Teknis yang Dipelajari

### 5.1 Game Loop dan Delta Time

`_process(delta)` digunakan untuk timer spawn, pembersihan referensi pipa, dan animasi latar. Pergerakan pipa menggunakan `_physics_process(delta)` agar perubahan posisi berbasis waktu, bukan jumlah frame.

Contoh:
```gdscript
position.x -= speed * delta
```

### 5.2 CharacterBody2D dan Gravitasi

Burung menggunakan CharacterBody2D. Kecepatan vertikal ditambah gravitasi pada setiap physics frame.

```gdscript
velocity.y = minf(velocity.y + gravity * delta, max_fall_speed)
move_and_slide()
```

Ketika pemain mengepak, velocity vertikal diberi impuls negatif sehingga burung bergerak ke atas.

### 5.3 Collision Layer dan Mask

- Bird: collision layer 1, collision mask 2, dengan bentuk collision radius 17.
- Pipe (AnimatableBody2D): collision layer 2, collision mask 0.
- Hazard Area2D: collision layer 0, collision mask 1.
- Score Area2D: collision layer 0, collision mask 1.

Bird mendeteksi pipe solid melalui mask 2. Hazard Area2D dan Score Area2D hanya mendeteksi Bird melalui layer 1; keduanya tidak menjadi penghalang fisik. Hazard Area2D mengikuti posisi pasangan pipa, sehingga overlap tetap terdeteksi ketika pipa bergerak ke arah burung.

### 5.4 Signal

PipePair mengirim signal `scored` ketika Bird masuk ke Score Area2D. Main menerima signal tersebut, menambahkan skor, memperbarui HUD, dan memeriksa target kemenangan.

Variabel `has_scored` menjaga agar satu pasangan pipa hanya memberikan satu poin.

PipePair juga mempunyai Hazard Area2D pada pipa atas dan bawah. Ketika Bird masuk ke area bahaya, PipePair mengirim signal `crashed`. Main menghubungkan signal tersebut ke `_on_bird_crashed()`, lalu mengakhiri run dan membuka Result Screen GAME OVER. Variabel `has_crashed` mencegah signal tabrakan dikirim berulang kali dari dua area pipa.

Bird tetap memiliki pemeriksaan slide collision sebagai lapisan perlindungan tambahan. Deteksi utama untuk pipa yang bergerak dilakukan oleh Hazard Area2D agar tidak bergantung hanya pada `get_slide_collision_count()`.

### 5.5 Koreksi Posisi Pipa

Koordinat `position` pada Node2D adalah titik tengah pasangan pipa. Karena ukuran pipa bawah adalah 760 piksel, posisi tengahnya harus dihitung sebagai:

```gdscript
var half_gap := gap_size * 0.5
var lower_pipe_center_y := half_gap + PIPE_LENGTH * 0.5
```

Dengan begitu, tepi atas pipa bawah tepat berada pada batas bawah celah. Kesalahan memakai `half_gap` sebagai titik tengah akan membuat pipa bawah masuk ke area celah dan membuat game sulit atau tidak mungkin dimainkan.

### 5.6 Autoload dan Penyimpanan Data

`game_data.gd` didaftarkan sebagai Autoload dengan nama `GameData`. Data skor dapat diakses lintas scene tanpa membawa node data secara manual.

Best score disimpan dengan ConfigFile di `user://savegame.cfg`. Lokasi `user://` dikelola oleh Godot dan terpisah dari folder source code proyek.

## 6. Parameter Balancing

| Parameter | File | Nilai awal | Fungsi |
|---|---|---:|---|
| gravity | bird.gd | 980 | Kekuatan gravitasi |
| flap_impulse | bird.gd | -360 | Impuls saat mengepak |
| max_fall_speed | bird.gd | 620 | Batas kecepatan jatuh |
| PIPE_INTERVAL | main.gd | 1.58 s | Jarak waktu spawn |
| BASE_PIPE_SPEED | main.gd | 188 | Kecepatan awal pipa |
| MIN_PIPE_GAP | main.gd | 154 | Celah minimum |
| TARGET_SCORE | game_data.gd | 10 | Target kemenangan |

Mahasiswa dapat mengubah nilai tersebut lalu membandingkan tingkat kesulitan permainan. Pastikan menguji perubahan satu parameter dalam satu waktu agar pengaruhnya dapat diamati.

## 7. Saran Praktikum dan Pengembangan

1. Ubah gravitasi dari 980 menjadi 800, kemudian 1.100. Bandingkan sensasi gerak.
2. Ubah impuls flap dari -360 menjadi -320 dan -400.
3. Ubah target skor dari 10 menjadi 15.
4. Tambahkan suara flap, skor, dan game over menggunakan AudioStreamPlayer.
5. Tambahkan animasi transisi scene.
6. Tambahkan animasi angka ketika skor bertambah.
7. Buat tabel pengujian dan minta minimal tiga orang mencoba game.
8. Dokumentasikan hasil balancing: parameter, nilai awal, nilai perubahan, dan pengaruh ke gameplay.

## 8. Peta ke RPS Pengantar Game

| Pertemuan | Materi | Bagian proyek |
|---|---|---|
| 3 | Game loop dan FSM | `_process`, `_physics_process`, GameState |
| 5 | Scene dan Node Godot | Main, Bird, PipePair, Menu, Result |
| 6 | Visual dan animasi sederhana | Polygon2D, gambar procedural, animasi sayap |
| 7 | Input, gerak dan collision | Input action, gravitasi, CharacterBody2D |
| 9 | Skor dan kondisi menang/kalah | Area2D, signal, GameData, Result |
| 10 | UI dan HUD | Menu, score, best score, pause |
| 12 | Balancing dan playtesting | Gravity, flap impulse, gap, speed |
| 14 | Build dan export | Export preset Godot |
| 15–16 | Prototipe dan demonstrasi | Integrasi keseluruhan game |

## 9. Menjalankan Smoke Test

Proyek menyediakan pengujian otomatis untuk memeriksa geometri celah pipa, signal Area2D (termasuk agar tidak memberi skor berulang), win condition, lose condition, dan perpindahan scene.

Jalankan dari terminal pada folder proyek:

```powershell
godot --headless --path . --script res://tests/smoke_test.gd
```

Jika semua pengujian berhasil, terminal menampilkan `ALL SMOKE TESTS PASSED`.

## 10. Checklist Sebelum Demonstrasi

- [ ] Main Menu muncul ketika game dijalankan.
- [ ] Tombol Mulai membuka gameplay.
- [ ] Input keyboard dan mouse bekerja.
- [ ] Celah antara pipa atas dan bawah terbentuk dengan benar.
- [ ] Collision pipa menyebabkan Game Over.
- [ ] Skor bertambah satu kali per pasangan pipa.
- [ ] Skor tidak bertambah setelah permainan selesai.
- [ ] ESC dan tombol JEDA dapat menghentikan permainan.
- [ ] Tombol LANJUTKAN mengaktifkan kembali permainan.
- [ ] Skor 10 menampilkan LEVEL CLEAR.
- [ ] Best score tetap tersimpan setelah aplikasi ditutup dan dibuka kembali.
- [ ] Restart dan kembali ke menu bekerja.

---

**Catatan dosen:** Proyek ini merupakan starting point pembelajaran. Mahasiswa sebaiknya diminta menggambar pohon scene, menjelaskan alur state, dan memodifikasi sekurang-kurangnya satu parameter balancing atau satu fitur sebelum melakukan demonstrasi.
