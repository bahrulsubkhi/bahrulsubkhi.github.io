extends Control
## Menu pembuka. Seluruh elemen dibangun dengan Control bawaan Godot.

const GAME_SCENE := "res://scenes/main.tscn"
var elapsed := 0.0

func _ready() -> void:
    set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    _build_ui()
    queue_redraw()

func _process(delta: float) -> void:
    elapsed += delta
    queue_redraw()

func _draw() -> void:
    var size := get_viewport_rect().size
    draw_rect(Rect2(Vector2.ZERO, size), Color("#79D5EE"))
    draw_rect(Rect2(0, size.y * 0.53, size.x, size.y * 0.47), Color("#C5F0DD"))
    draw_circle(Vector2(size.x * 0.79, size.y * 0.15), 38.0, Color("#FFE59A"))
    _draw_cloud(Vector2(80 + sin(elapsed * 0.15) * 8, 130), 0.9)
    _draw_cloud(Vector2(370 - sin(elapsed * 0.13) * 10, 205), 0.72)
    _draw_cloud(Vector2(240, 75), 0.55)
    draw_colored_polygon(PackedVector2Array([
        Vector2(0, 500), Vector2(75, 435), Vector2(160, 490),
        Vector2(255, 410), Vector2(365, 485), Vector2(480, 425),
        Vector2(480, 720), Vector2(0, 720)
    ]), Color("#8CCB9D"))
    draw_colored_polygon(PackedVector2Array([
        Vector2(0, 570), Vector2(105, 510), Vector2(220, 565),
        Vector2(340, 500), Vector2(480, 555), Vector2(480, 720),
        Vector2(0, 720)
    ]), Color("#5FAF83"))

func _draw_cloud(center: Vector2, scale_factor: float) -> void:
    var cloud_color := Color("#F7FFFF", 0.86)
    draw_circle(center, 22.0 * scale_factor, cloud_color)
    draw_circle(center + Vector2(22, -9) * scale_factor,
        27.0 * scale_factor, cloud_color)
    draw_circle(center + Vector2(47, 1) * scale_factor,
        19.0 * scale_factor, cloud_color)
    draw_rect(Rect2(center + Vector2(-5, 0) * scale_factor,
        Vector2(55, 18) * scale_factor), cloud_color)

func _build_ui() -> void:
    var layout := VBoxContainer.new()
    layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    layout.offset_left = 38
    layout.offset_right = -38
    layout.offset_top = 28
    layout.offset_bottom = -28
    layout.alignment = BoxContainer.ALIGNMENT_CENTER
    layout.add_theme_constant_override("separation", 10)
    add_child(layout)

    layout.add_child(_make_label("UPJB  •  INFORMATIKA", 15, Color("#16465D")))

    var title := _make_label("FLAPPY\nBIRD", 66, Color("#123D54"))
    title.add_theme_color_override("font_shadow_color", Color("#FFFFFF", 0.45))
    title.add_theme_constant_override("shadow_offset_x", 2)
    title.add_theme_constant_override("shadow_offset_y", 4)
    title.custom_minimum_size = Vector2(0, 148)
    layout.add_child(title)

    layout.add_child(_make_label(
        "TERBANG, HINDARI PIPA, RAIH 10 POIN!", 14, Color("#20566A")
    ))

    var spacer := Control.new()
    spacer.custom_minimum_size = Vector2(0, 8)
    layout.add_child(spacer)

    var info := PanelContainer.new()
    info.add_theme_stylebox_override("panel",
        _panel_style(Color("#FFFFFF"), Color("#FFFFFF", 0.55)))
    info.custom_minimum_size = Vector2(0, 126)
    layout.add_child(info)

    var info_box := VBoxContainer.new()
    info_box.alignment = BoxContainer.ALIGNMENT_CENTER
    info_box.add_theme_constant_override("separation", 8)
    info.add_child(info_box)
    info_box.add_child(_make_label("CARA BERMAIN", 18, Color("#17475B")))

    var instructions := _make_label(
        "Tekan SPACE / W / PANAH ATAS atau klik kiri untuk mengepak.\nLewati celah pipa dan hindari rintangan.",
        13, Color("#37677A")
    )
    instructions.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    instructions.custom_minimum_size = Vector2(0, 48)
    info_box.add_child(instructions)

    var play := _make_button("MULAI BERMAIN", true)
    play.pressed.connect(_on_play_pressed)
    layout.add_child(play)

    layout.add_child(_make_label(
        "TARGET 10 POIN  •  BEST SCORE TERSIMPAN OTOMATIS",
        11, Color("#24566B")
    ))

func _make_label(value: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = value
    label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    label.add_theme_color_override("font_shadow_color", Color("#FFFFFF", 0.35))
    label.add_theme_constant_override("shadow_offset_y", 1)
    return label

func _make_button(value: String, primary: bool) -> Button:
    var button := Button.new()
    button.text = value
    button.custom_minimum_size = Vector2(0, 56)
    button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    button.add_theme_font_size_override("font_size", 18)
    button.add_theme_color_override("font_color", Color("#143F50"))
    var normal := StyleBoxFlat.new()
    normal.bg_color = Color("#FFD36E") if primary else Color("#E7F7F8")
    normal.set_corner_radius_all(18)
    normal.shadow_color = Color("#17475B", 0.2)
    normal.shadow_size = 8
    normal.shadow_offset = Vector2(0, 4)
    var hover := normal.duplicate() as StyleBoxFlat
    hover.bg_color = Color("#FFE39A") if primary else Color("#FFFFFF")
    var pressed := normal.duplicate() as StyleBoxFlat
    pressed.bg_color = Color("#F2B94E") if primary else Color("#CDECEF")
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
    style.set_corner_radius_all(22)
    style.content_margin_left = 20
    style.content_margin_right = 20
    style.content_margin_top = 14
    style.content_margin_bottom = 14
    style.shadow_color = Color("#16465D", 0.18)
    style.shadow_size = 12
    style.shadow_offset = Vector2(0, 5)
    return style

func _on_play_pressed() -> void:
    GameData.start_run()
    get_tree().change_scene_to_file(GAME_SCENE)

func _unhandled_input(event: InputEvent) -> void:
    if event.is_action_pressed("flap"):
        _on_play_pressed()

