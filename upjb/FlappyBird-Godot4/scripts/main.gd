extends Node2D
## Game loop, finite state machine, spawn rintangan, skor dan HUD.
enum GameState { READY, PLAYING, GAME_OVER }
@export var bird_scene: PackedScene
@export var pipe_scene: PackedScene
const SCREEN_SIZE := Vector2(480, 720)
const PIPE_INTERVAL := 1.55
const BASE_PIPE_SPEED := 185.0
var state: GameState = GameState.READY
var score := 0
var best_score := 0
var spawn_timer := 0.0
var pipe_speed := BASE_PIPE_SPEED
var rng := RandomNumberGenerator.new()
var bird: CharacterBody2D
var pipes: Array[Node2D] = []
var score_label: Label
var best_label: Label
var message_label: Label
var hint_label: Label
var restart_button: Button
var pause_button: Button
var paused := false
func _ready() -> void:
	rng.randomize()
	_build_world()
	_build_hud()
	_load_best_score()
	_reset_round()
	_change_state(GameState.READY)
func _process(delta: float) -> void:
	# Game loop: update berjalan setiap frame dan memakai delta time.
	if state != GameState.PLAYING or paused:
		return
	spawn_timer += delta
	if spawn_timer >= PIPE_INTERVAL:
		spawn_timer = 0.0
		_spawn_pipe()
	for pipe in pipes.duplicate():
		if not is_instance_valid(pipe) or pipe.position.x < -120.0:
			pipes.erase(pipe)
	_update_hud()
func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("flap") or (event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT):
		if state == GameState.READY:
			_change_state(GameState.PLAYING)
			bird.flap()
		elif state == GameState.PLAYING and not paused:
			bird.flap()
		elif state == GameState.GAME_OVER:
			_start_game()
	elif event.is_action_pressed("ui_cancel") and state == GameState.PLAYING:
		_toggle_pause()
func _physics_process(_delta: float) -> void:
	if state != GameState.PLAYING or paused:
		return
	if bird.global_position.y < 18.0 or bird.global_position.y > 650.0:
		_game_over()
func _build_world() -> void:
	var background := ColorRect.new()
	background.color = Color("#83d7f2")
	background.size = SCREEN_SIZE
	background.z_index = -20
	add_child(background)
	var hill := Polygon2D.new()
	hill.color = Color("#8bd17c")
	hill.polygon = PackedVector2Array([Vector2(0, 535), Vector2(70, 490), Vector2(155, 525), Vector2(245, 475), Vector2(350, 520), Vector2(480, 480), Vector2(480, 680), Vector2(0, 680)])
	hill.z_index = -5
	add_child(hill)
	var ground := Polygon2D.new()
	ground.color = Color("#e8c878")
	ground.polygon = PackedVector2Array([Vector2(0, 660), Vector2(480, 660), Vector2(480, 720), Vector2(0, 720)])
	add_child(ground)
	var ground_line := Line2D.new()
	ground_line.points = PackedVector2Array([Vector2(0, 660), Vector2(480, 660)])
	ground_line.width = 7.0
	ground_line.default_color = Color("#6fb84e")
	add_child(ground_line)
	bird = bird_scene.instantiate() as CharacterBody2D
	bird.position = Vector2(140, 330)
	bird.crashed.connect(_on_bird_crashed)
	add_child(bird)
func _build_hud() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	score_label = Label.new()
	score_label.position = Vector2(210, 30)
	score_label.size = Vector2(120, 62)
	score_label.add_theme_font_size_override("font_size", 48)
	score_label.add_theme_color_override("font_color", Color.WHITE)
	score_label.add_theme_color_override("font_shadow_color", Color("#27536b"))
	score_label.add_theme_constant_override("shadow_offset_x", 3)
	score_label.add_theme_constant_override("shadow_offset_y", 3)
	layer.add_child(score_label)
	best_label = Label.new()
	best_label.position = Vector2(18, 18)
	best_label.add_theme_font_size_override("font_size", 16)
	best_label.add_theme_color_override("font_color", Color("#17465a"))
	layer.add_child(best_label)
	message_label = Label.new()
	message_label.position = Vector2(35, 210)
	message_label.size = Vector2(410, 130)
	message_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	message_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	message_label.add_theme_font_size_override("font_size", 29)
	message_label.add_theme_color_override("font_color", Color("#17465a"))
	layer.add_child(message_label)
	hint_label = Label.new()
	hint_label.position = Vector2(45, 360)
	hint_label.size = Vector2(390, 50)
	hint_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	hint_label.add_theme_font_size_override("font_size", 16)
	hint_label.add_theme_color_override("font_color", Color("#17465a"))
	layer.add_child(hint_label)
	pause_button = Button.new()
	pause_button.text = "Jeda"
	pause_button.position = Vector2(390, 18)
	pause_button.size = Vector2(72, 36)
	pause_button.pressed.connect(_toggle_pause)
	layer.add_child(pause_button)
	restart_button = Button.new()
	restart_button.text = "Main Lagi"
	restart_button.position = Vector2(155, 405)
	restart_button.size = Vector2(170, 48)
	restart_button.pressed.connect(_start_game)
	layer.add_child(restart_button)
func _change_state(next_state: GameState) -> void:
	# FSM: READY -> PLAYING -> GAME_OVER.
	state = next_state
	match state:
		GameState.READY:
			bird.stop()
			message_label.text = "FLAPPY BIRD\nPENGANTAR GAME"
			hint_label.text = "Tekan SPACE / W / klik untuk mulai"
			restart_button.visible = false
			pause_button.visible = false
		GameState.PLAYING:
			bird.start()
			message_label.text = ""
			hint_label.text = ""
			restart_button.visible = false
			pause_button.visible = true
		GameState.GAME_OVER:
			bird.stop()
			message_label.text = "GAME OVER"
			hint_label.text = "Skor: %d     Best: %d" % [score, best_score]
			restart_button.visible = true
			pause_button.visible = false
func _start_game() -> void:
	_reset_round()
	_change_state(GameState.PLAYING)
	bird.flap()
func _reset_round() -> void:
	score = 0
	spawn_timer = 0.0
	pipe_speed = BASE_PIPE_SPEED
	paused = false
	for pipe in pipes:
		if is_instance_valid(pipe):
			pipe.queue_free()
	pipes.clear()
	bird.stop()
	bird.position = Vector2(140, 330)
	bird.rotation = 0.0
	restart_button.visible = false
	pause_button.text = "Jeda"
	_update_hud()
func _spawn_pipe() -> void:
	var pipe := pipe_scene.instantiate() as Node2D
	pipe.position = Vector2(535, rng.randf_range(220.0, 440.0))
	pipe.speed = pipe_speed
	pipe.scored.connect(_on_pipe_scored)
	add_child(pipe)
	pipes.append(pipe)
func _on_pipe_scored() -> void:
	if state != GameState.PLAYING:
		return
	score += 1
	pipe_speed = minf(BASE_PIPE_SPEED + float(score) * 3.5, 285.0)
	if score > best_score:
		best_score = score
		_save_best_score()
	_update_hud()
func _on_bird_crashed() -> void:
	_game_over()
func _game_over() -> void:
	if state != GameState.PLAYING:
		return
	_change_state(GameState.GAME_OVER)
func _toggle_pause() -> void:
	if state != GameState.PLAYING:
		return
	paused = not paused
	bird.can_move = not paused
	for pipe in pipes:
		if is_instance_valid(pipe):
			pipe.set_physics_process(not paused)
	pause_button.text = "Lanjut" if paused else "Jeda"
	message_label.text = "PAUSE" if paused else ""
	hint_label.text = "Tekan ESC untuk lanjut" if paused else ""
func _update_hud() -> void:
	score_label.text = str(score)
	best_label.text = "BEST: %d" % best_score
func _load_best_score() -> void:
	if FileAccess.file_exists("user://savegame.cfg"):
		var config := ConfigFile.new()
		if config.load("user://savegame.cfg") == OK:
			best_score = int(config.get_value("score", "best", 0))
func _save_best_score() -> void:
	var config := ConfigFile.new()
	config.set_value("score", "best", best_score)
	config.save("user://savegame.cfg")
