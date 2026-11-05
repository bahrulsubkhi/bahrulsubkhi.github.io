# Flappy Bird — Godot 4
**Proyek demonstrasi mata kuliah Pengantar Game · S1 Informatika UPJB · Semester 3**

Game 2D playable dengan GDScript dan node bawaan Godot. Semua visual burung dan pipa dibuat secara procedural, tanpa asset eksternal.

## Menjalankan
1. Buka Godot 4.7.2 atau Godot 4.x.
2. Pilih **Import** lalu pilih file `project.godot` pada folder ini.
3. Tekan **F6/F5** atau tombol Play.
4. Tekan **SPACE**, **W**, **panah atas**, atau klik mouse untuk mengepak.
5. Tombol **Jeda** atau ESC untuk pause; klik Main Lagi untuk restart.

## Struktur
- `project.godot` — konfigurasi project, resolusi dan input action.
- `scenes/main.tscn` — root scene dan referensi scene lain.
- `scenes/bird.tscn` — node karakter dan collision shape.
- `scenes/pipe_pair.tscn` — scene rintangan.
- `scripts/main.gd` — game loop, FSM, spawn, skor, HUD, pause, high score.
- `scripts/bird.gd` — gravitasi, input, velocity, gerak dan collision.
- `scripts/pipe_pair.gd` — pipa procedural, physics body dan Area2D skor.

## Peta ke RPS
| Minggu | Materi | Implementasi |
|---|---|---|
| 3 | Game loop dan FSM | `_process(delta)`, `_physics_process`, `GameState` |
| 5 | Scene dan Node Godot | Main, Bird, PipePair, Node2D, CharacterBody2D |
| 6 | Visual/animasi sederhana | Polygon2D dan rotasi mengikuti velocity |
| 7 | Input, gerak dan collision | InputEvent, gravitasi, `move_and_slide`, collision layer |
| 9 | Skor dan kondisi kalah | `Area2D.body_entered`, signal `scored`, Game Over |
| 10 | UI dan HUD | Label skor, best score, tombol jeda/restart |
| 12 | Balancing/playtesting | gravity, flap_impulse, gap_size, PIPE_INTERVAL, kecepatan |
| 14 | Build/export | Export Windows/Linux/Web melalui preset Godot |
| 15–16 | Prototipe dan demo | Integrasi alur READY → PLAYING → GAME_OVER |

## Konsep utama
- **Game loop:** `_process(delta)` memperbarui timer dan objek. Delta time menjaga gerakan relatif konsisten terhadap frame rate.
- **Finite State Machine:** `READY`, `PLAYING`, dan `GAME_OVER` mengontrol perilaku dan UI.
- **Physics:** gravity menambah kecepatan vertikal; flap_impulse memberi impuls ke atas.
- **Collision:** CharacterBody2D bertabrakan dengan StaticBody2D; Area2D menjadi trigger skor.
- **Signals:** `crashed` dan `scored` memisahkan komunikasi antar node.
- **Balancing:** kecepatan pipa meningkat perlahan setiap skor.
- **Persistence:** high score disimpan ke `user://savegame.cfg`.

## Praktikum bertahap
1. Ubah `gravity` dan `flap_impulse`, lalu catat efeknya.
2. Ubah `gap_size` dan `PIPE_INTERVAL` untuk mengatur kesulitan.
3. Tambahkan animasi sayap memakai AnimationPlayer.
4. Tambahkan suara flap, score, dan game over memakai AudioStreamPlayer.
5. Tambahkan scene menu terpisah dan transisi scene.
6. Playtest minimal 3 pemain; catat feedback dan balancing.

## Catatan dosen
Minta mahasiswa menggambar pohon scene dan diagram FSM sebelum membaca script. Lanjutkan dengan modifikasi parameter dan fitur kecil agar mereka memahami hubungan node, script, signal, dan gameplay. Project ini merupakan starting point pembelajaran, bukan template final tanpa modifikasi.
