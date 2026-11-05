extends Node2D
## Gameplay: environment, spawn, HUD, FSM, scoring and pause controls.

enum GameState { READY, PLAYING, PAUSED, FINISHED }

@export var bird_scene: PackedScene
@export var pipe_scene: PackedScene

const SCREEN_SIZE := Vector2(480, 720)
const PIPE_INTERVAL := 1.58
const BASE_PIPE_SPEED := 188.0
const MIN_PIPE_GAP := 154.0

var state: GameState = GameState.READY
var spawn_timer := 0.0
var elapsed := 0.0
var pipe_speed := BASE_PIPE_SPEED
var rng := RandomNumberGenerator.new()
var bird: Bird
var pipes: Array[PipePair] = []

var score_label: Label
var best_label: Label
var pause_button: Button
var ready_panel: PanelContainer
var pause_panel: PanelContainer

func _ready() -> void:
    rng.randomize()
    GameData.start_run()
    _create_bird()
    _build_hud()
    _set_state(GameState.READY)
    queue_redraw()

func _process(delta: float) -> void:
    elapsed += delta
    queue_redraw()

    if state != GameState.PLAYING:
        return

    spawn_timer += delta
    if spawn_timer >= PIPE_INTERVAL:
        spawn_timer -= PIPE_INTERVAL
        _spawn_pipe()

    for index in range(pipes.size() - 1, -1, -1):
        if not is_instance_valid(pipes[index]):
            pipes.remove_at(index)

func _physics_process(_delta: float) -> void:
    if state != GameState.PLAYING:
        return
    if bird.position.y < -24.0 or bird.position.y > 636.0:
        _finish_run(false)

func _unhandled_input(event: InputEvent) -> void:
    if event.is_action_pressed("ui_cancel"):
        if state == GameState.PLAYING:
            _pause_game()
        elif state == GameState.PAUSED:
            _resume_game()
        get_viewport().set_input_as_handled()
        return

    var mouse_flap := false
    if event is InputEventMouseButton:
        mouse_flap = event.pressed and event.button_index == MOUSE_BUTTON_LEFT

    if event.is_action_pressed("flap") or mouse_flap:
        if state == GameState.READY:
            _start_playing()
        elif state == GameState.PLAYING:
            bird.flap()
        get_viewport().set_input_as_handled()

func _draw() -> void:
    draw_rect(Rect2(Vector2.ZERO, SCREEN_SIZE), Color("#81D8F1"))
    draw_rect(Rect2(Vector2(0, 470), Vector2(480, 250)), Color("#BDEEDC"))
    draw_circle(Vector2(390, 112), 35.0, Color("#FFE59A"))
    _draw_cloud(Vector2(74 + sin(elapsed * 0.13) * 7, 155), 0.9)
    _draw_cloud(Vector2(300 - sin(elapsed * 0.1) * 9, 240), 0.72)
    _draw_cloud(Vector2(220, 72), 0.52)

    draw_colored_polygon(PackedVector2Array([
        Vector2(0, 535), Vector2(74, 470), Vector2(160, 520),
        Vector2(250, 445), Vector2(360, 510), Vector2(480, 450),
        Vector2(480, 660), Vector2(0, 660)
    ]), Color("#93CFA2"))
    draw_colored_polygon(PackedVector2Array([
        Vector2(0, 590), Vector2(100, 530), Vector2(210, 580),
        Vector2(340, 520), Vector2(480, 575), Vector2(480, 660),
        Vector2(0, 660)
    ]), Color("#65B584"))
    draw_rect(Rect2(0, 650, 480, 70), Color("#EBCB83"))
    draw_rect(Rect2(0, 650, 480, 11), Color("#58B96C"))
    for x in range(0, 520, 48):
        draw_line(Vector2(x, 674), Vector2(x + 25, 674), Color("#D9B76D"), 3.0)
        draw_line(Vector2(x + 17, 699), Vector2(x + 38, 699), Color("#D9B76D"), 2.0)

func _draw_cloud(center: Vector2, scale_factor: float) -> void:
    var cloud_color := Color("#F8FFFF", 0.88)
    draw_circle(center, 20.0 * scale_factor, cloud_color)
    draw_circle(center + Vector2(22, -9) * scale_factor,
        26.0 * scale_factor, cloud_color)
    draw_circle(center + Vector2(46, 1) * scale_factor,
        18.0 * scale_factor, cloud_color)
    draw_rect(Rect2(center + Vector2(-4, 0) * scale_factor,
        Vector2(53, 17) * scale_factor), cloud_color)

func _create_bird() -> void:
    bird = bird_scene.instantiate() as Bird
    bird.position = Vector2(142, 334)
    bird.crashed.connect(_on_bird_crashed)
    add_child(bird)

func _build_hud() -> void:
    var layer := CanvasLayer.new()
    add_child(layer)

    var top_bar := PanelContainer.new()
    top_bar.position = Vector2(14, 13)
    top_bar.size = Vector2(452, 66)
    top_bar.add_theme_stylebox_override("panel",
        _panel_style(Color("#FFFFFF", 0.91), Color("#FFFFFF", 0.65)))
    layer.add_child(top_bar)

    var row := HBoxContainer.new()
    row.add_theme_constant_override("separation", 8)
    top_bar.add_child(row)

    best_label = Label.new()
    best_label.text = "BEST  00"
    best_label.custom_minimum_size = Vector2(100, 0)
    best_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
    best_label.add_theme_font_size_override("font_size", 15)
    best_label.add_theme_color_override("font_color", Color("#24556A"))
    row.add_child(best_label)

    score_label = Label.new()
    score_label.text = "0"
    score_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    score_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    score_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
    score_label.add_theme_font_size_override("font_size", 34)
    score_label.add_theme_color_override("font_color", Color("#123F55"))
    row.add_child(score_label)

    pause_button = _new_button("JEDA", false)
    pause_button.custom_minimum_size = Vector2(82, 42)
    pause_button.pressed.connect(_toggle_pause)
    row.add_child(pause_button)

    var ready_data := _make_overlay(
        "SIAP TERBANG?",
        "Lewati celah di antara pipa.\nCapai 10 poin untuk memenangkan permainan."
    )
    ready_panel = ready_data["panel"]
    var ready_content: VBoxContainer = ready_data["content"]
    var start_button := _new_button("MULAI TERBANG", true)
    start_button.pressed.connect(_start_playing)
    ready_content.add_child(start_button)
    ready_content.add_child(_make_small_label("SPACE  •  W  •  PANAH ATAS  •  KLIK KIRI"))
    layer.add_child(ready_panel)

    var pause_data := _make_overlay(
        "PERMAINAN DIJEDA",
        "Tarik napas sebentar.\nSiap melanjutkan perjalanan?"
    )
    pause_panel = pause_data["panel"]
    var pause_content: VBoxContainer = pause_data["content"]

    var resume_button := _new_button("LANJUTKAN", true)
    resume_button.pressed.connect(_resume_game)
    pause_content.add_child(resume_button)

    var restart_button := _new_button("ULANGI PERMAINAN", false)
    restart_button.pressed.connect(_restart_game)
    pause_content.add_child(restart_button)

    var menu_button := _new_button("MENU UTAMA", false)
    menu_button.pressed.connect(_go_to_menu)
    pause_content.add_child(menu_button)
    layer.add_child(pause_panel)

    pause_panel.hide()
    _update_hud()

func _make_overlay(title_text: String, description: String) -> Dictionary:
    var panel := PanelContainer.new()
    panel.position = Vector2(32, 183)
    panel.size = Vector2(416, 354)
    panel.add_theme_stylebox_override("panel",
        _panel_style(Color("#FFFFFF", 0.96), Color("#FFFFFF", 0.8)))

    var content := VBoxContainer.new()
    content.alignment = BoxContainer.ALIGNMENT_CENTER
    content.add_theme_constant_override("separation", 12)
    panel.add_child(content)

    var heading := Label.new()
    heading.text = title_text
    heading.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    heading.add_theme_font_size_override("font_size", 27)
    heading.add_theme_color_override("font_color", Color("#16465B"))
    content.add_child(heading)

    var body := Label.new()
    body.text = description
    body.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    body.custom_minimum_size = Vector2(0, 56)
    body.add_theme_font_size_override("font_size", 14)
    body.add_theme_color_override("font_color", Color("#426F7F"))
    content.add_child(body)

    return {"panel": panel, "content": content}

func _make_small_label(value: String) -> Label:
    var label := Label.new()
    label.text = value
    label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    label.add_theme_font_size_override("font_size", 11)
    label.add_theme_color_override("font_color", Color("#557D8A"))
    return label

func _new_button(value: String, primary: bool) -> Button:
    var button := Button.new()
    button.text = value
    button.custom_minimum_size = Vector2(0, 48)
    button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    button.add_theme_font_size_override("font_size", 15)
    button.add_theme_color_override("font_color", Color("#16465B"))
    var normal := StyleBoxFlat.new()
    normal.bg_color = Color("#FFD36E") if primary else Color("#E8F5F7")
    normal.set_corner_radius_all(15)
    var hover := normal.duplicate() as StyleBoxFlat
    hover.bg_color = Color("#FFE49C") if primary else Color("#FFFFFF")
    var pressed := normal.duplicate() as StyleBoxFlat
    pressed.bg_color = Color("#F0B84C") if primary else Color("#CDE8EC")
    button.add_theme_stylebox_override("normal", normal)
    button.add_theme_stylebox_override("hover", hover)
    button.add_theme_stylebox_override("pressed", pressed)
    button.add_theme_stylebox_override("focus", StyleBoxEmpty.new())
    return button

func _panel_style(background: Color, border: Color) -> StyleBoxFlat:
    var style := StyleBoxFlat.new()
    style.bg_color = background
    style.border_color = border
    style.set_border_width_all(2)
    style.set_corner_radius_all(21)
    style.content_margin_left = 20
    style.content_margin_right = 20
    style.content_margin_top = 16
    style.content_margin_bottom = 16
    style.shadow_color = Color("#16465D", 0.2)
    style.shadow_size = 12
    style.shadow_offset = Vector2(0, 5)
    return style

func _set_state(next_state: GameState) -> void:
    state = next_state
    ready_panel.visible = state == GameState.READY
    pause_panel.visible = state == GameState.PAUSED
    pause_button.visible = state == GameState.PLAYING
    _update_hud()

func _start_playing() -> void:
    if state != GameState.READY:
        return
    _set_state(GameState.PLAYING)
    bird.start()
    bird.flap()

func _pause_game() -> void:
    if state != GameState.PLAYING:
        return
    _set_state(GameState.PAUSED)
    bird.can_move = false
    for pipe in pipes:
        if is_instance_valid(pipe):
            pipe.set_physics_process(false)

func _resume_game() -> void:
    if state != GameState.PAUSED:
        return
    _set_state(GameState.PLAYING)
    bird.can_move = true
    for pipe in pipes:
        if is_instance_valid(pipe):
            pipe.set_physics_process(true)

func _toggle_pause() -> void:
    if state == GameState.PLAYING:
        _pause_game()
    elif state == GameState.PAUSED:
        _resume_game()

func _spawn_pipe() -> void:
    var obstacle := pipe_scene.instantiate() as PipePair
    obstacle.position = Vector2(535, rng.randf_range(220.0, 420.0))
    obstacle.speed = pipe_speed
    obstacle.gap_size = maxf(MIN_PIPE_GAP,
        194.0 - float(GameData.score) * 3.0)
    obstacle.scored.connect(_on_pipe_scored)
    obstacle.crashed.connect(_on_bird_crashed)
    add_child(obstacle)
    pipes.append(obstacle)

func _on_pipe_scored() -> void:
    if state != GameState.PLAYING:
        return
    GameData.add_point()
    pipe_speed = minf(BASE_PIPE_SPEED + float(GameData.score) * 4.0, 270.0)
    _update_hud()
    if GameData.score >= GameData.TARGET_SCORE:
        _finish_run(true)

func _on_bird_crashed() -> void:
    _finish_run(false)

func _finish_run(is_won: bool) -> void:
    if state == GameState.FINISHED:
        return
    _set_state(GameState.FINISHED)
    bird.stop()
    for pipe in pipes:
        if is_instance_valid(pipe):
            pipe.set_physics_process(false)
    GameData.complete_run(is_won)
    call_deferred("_open_result_scene")

func _open_result_scene() -> void:
    get_tree().change_scene_to_file("res://scenes/result.tscn")

func _restart_game() -> void:
    get_tree().change_scene_to_file("res://scenes/main.tscn")

func _go_to_menu() -> void:
    get_tree().change_scene_to_file("res://scenes/menu.tscn")

func _update_hud() -> void:
    if is_instance_valid(score_label):
        score_label.text = str(GameData.score)
    if is_instance_valid(best_label):
        best_label.text = "BEST  %02d" % GameData.best_score
