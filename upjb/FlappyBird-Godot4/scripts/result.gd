extends Control
## Scene hasil yang menampilkan kondisi menang atau kalah.

const MENU_SCENE := "res://scenes/menu.tscn"
const GAME_SCENE := "res://scenes/main.tscn"

func _ready() -> void:
    set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    _build_ui()
    queue_redraw()

func _draw() -> void:
    var size := get_viewport_rect().size
    draw_rect(Rect2(Vector2.ZERO, size), Color("#78D2EC"))
    draw_rect(Rect2(0, size.y * 0.56, size.x, size.y * 0.44), Color("#BDEEDC"))
    draw_circle(Vector2(size.x * 0.82, size.y * 0.15), 39.0, Color("#FFE59A"))
    draw_colored_polygon(PackedVector2Array([
        Vector2(0, 520), Vector2(90, 455), Vector2(180, 515),
        Vector2(300, 450), Vector2(480, 505), Vector2(480, 720),
        Vector2(0, 720)
    ]), Color("#75BA8C"))

    var colors := [
        Color("#FFD36E"), Color("#FFFFFF"),
        Color("#62C9AC"), Color("#FF9D75")
    ]
    for index in range(28):
        var x := fposmod(float(index * 67 + 31), size.x)
        var y := fposmod(float(index * 113 + 29), size.y * 0.8)
        draw_circle(Vector2(x, y), 3.0 + float(index % 3),
            colors[index % colors.size()])

func _build_ui() -> void:
    var layout := VBoxContainer.new()
    layout.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    layout.offset_left = 34
    layout.offset_right = -34
    layout.offset_top = 42
    layout.offset_bottom = -42
    layout.alignment = BoxContainer.ALIGNMENT_CENTER
    layout.add_theme_constant_override("separation", 14)
    add_child(layout)

    layout.add_child(_make_label(
        "HASIL PERMAINAN", 14, Color("#1D5268")
    ))

    var title_text := "LEVEL CLEAR!" if GameData.won else "GAME OVER"
    var title_color := Color("#176A58") if GameData.won else Color("#9B4B3D")
    var title := _make_label(title_text, 42, title_color)
    title.custom_minimum_size = Vector2(0, 82)
    layout.add_child(title)

    var subtitle := "Belum berhasil? Coba lagi dan lewati semua pipa."
    if GameData.won:
        subtitle = "Hebat! Target 10 poin berhasil dicapai."
    layout.add_child(_make_label(subtitle, 13, Color("#315F70")))

    var card := PanelContainer.new()
    card.custom_minimum_size = Vector2(0, 148)
    card.add_theme_stylebox_override("panel",
        _panel_style(Color("#FFFFFF", 0.94), Color("#FFFFFF", 0.72)))
    layout.add_child(card)

    var card_box := VBoxContainer.new()
    card_box.alignment = BoxContainer.ALIGNMENT_CENTER
    card_box.add_theme_constant_override("separation", 8)
    card.add_child(card_box)
    card_box.add_child(_make_label("SKOR AKHIR", 13, Color("#527787")))
    card_box.add_child(_make_label(str(GameData.score), 54, Color("#17475B")))
    card_box.add_child(_make_label(
        "BEST SCORE  %02d" % GameData.best_score, 15, Color("#34705F")
    ))

    var restart := _make_button("MAIN LAGI", true)
    restart.pressed.connect(_on_restart_pressed)
    layout.add_child(restart)

    var menu := _make_button("KEMBALI KE MENU", false)
    menu.pressed.connect(_on_menu_pressed)
    layout.add_child(menu)

    layout.add_child(_make_label(
        "FLAPPY BIRD  •  PENGANTAR GAME  •  UPJB",
        10, Color("#38677A")
    ))

func _make_label(value: String, size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = value
    label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
    label.add_theme_font_size_override("font_size", size)
    label.add_theme_color_override("font_color", color)
    return label

func _make_button(value: String, primary: bool) -> Button:
    var button := Button.new()
    button.text = value
    button.custom_minimum_size = Vector2(0, 54)
    button.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    button.add_theme_font_size_override("font_size", 16)
    button.add_theme_color_override("font_color", Color("#16465B"))
    var normal := StyleBoxFlat.new()
    normal.bg_color = Color("#FFD36E") if primary else Color("#E8F5F7")
    normal.set_corner_radius_all(16)
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
    style.set_corner_radius_all(22)
    style.content_margin_left = 22
    style.content_margin_right = 22
    style.content_margin_top = 16
    style.content_margin_bottom = 16
    style.shadow_color = Color("#16465D", 0.18)
    style.shadow_size = 12
    style.shadow_offset = Vector2(0, 5)
    return style

func _on_restart_pressed() -> void:
    GameData.start_run()
    get_tree().change_scene_to_file(GAME_SCENE)

func _on_menu_pressed() -> void:
    get_tree().change_scene_to_file(MENU_SCENE)

