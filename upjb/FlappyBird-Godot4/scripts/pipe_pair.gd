extends Node2D
class_name PipePair
## Rintangan dua pipa, collision solid, dan satu trigger skor.

signal scored
signal crashed

@export var speed: float = 185.0
@export var gap_size: float = 194.0
@export var pipe_width: float = 74.0

var has_scored := false
var has_crashed := false

const PIPE_LENGTH := 760.0
const CAP_HEIGHT := 25.0

func _ready() -> void:
    # Posisi Node2D adalah titik tengah body. Kedua pipa harus
    # diletakkan menjauh dari pusat sesuai setengah panjang body.
    var half_gap := gap_size * 0.5
    _build_pipe(
        Vector2(0, -half_gap - PIPE_LENGTH * 0.5),
        Vector2(pipe_width, PIPE_LENGTH), 1.0
    )
    _build_pipe(
        Vector2(0, half_gap + PIPE_LENGTH * 0.5),
        Vector2(pipe_width, PIPE_LENGTH), -1.0
    )
    _build_score_area()

func _physics_process(delta: float) -> void:
    position.x -= speed * delta
    if position.x < -150.0:
        queue_free()

func _build_pipe(center: Vector2, size: Vector2, cap_direction: float) -> void:
    var body := AnimatableBody2D.new()
    body.position = center
    body.collision_layer = 2
    body.collision_mask = 0
    body.sync_to_physics = true

    var half_w := size.x * 0.5
    var half_h := size.y * 0.5

    var visual := Polygon2D.new()
    visual.polygon = PackedVector2Array([
        Vector2(-half_w, -half_h), Vector2(half_w, -half_h),
        Vector2(half_w, half_h), Vector2(-half_w, half_h)
    ])
    visual.color = Color("#32AE67")
    body.add_child(visual)

    var highlight := Polygon2D.new()
    highlight.position = Vector2(-half_w + 10, 0)
    highlight.polygon = PackedVector2Array([
        Vector2(0, -half_h), Vector2(8, -half_h),
        Vector2(8, half_h), Vector2(0, half_h)
    ])
    highlight.color = Color("#70D78C")
    body.add_child(highlight)

    var shade := Polygon2D.new()
    shade.position = Vector2(half_w - 9, 0)
    shade.polygon = PackedVector2Array([
        Vector2(-3, -half_h), Vector2(0, -half_h),
        Vector2(0, half_h), Vector2(-3, half_h)
    ])
    shade.color = Color("#228B56")
    body.add_child(shade)

    var collider := CollisionShape2D.new()
    var shape := RectangleShape2D.new()
    shape.size = size
    collider.shape = shape
    body.add_child(collider)

    var cap_y := cap_direction * size.y * 0.5
    _add_pipe_cap(body, cap_y)
    add_child(body)
    _build_hazard_area(center, size, cap_direction)

func _add_pipe_cap(parent: AnimatableBody2D, local_y: float) -> void:
    var cap_width := pipe_width + 14.0
    var cap := Polygon2D.new()
    cap.position = Vector2(0, local_y)
    cap.polygon = PackedVector2Array([
        Vector2(-cap_width * 0.5, -CAP_HEIGHT * 0.5),
        Vector2(cap_width * 0.5, -CAP_HEIGHT * 0.5),
        Vector2(cap_width * 0.5, CAP_HEIGHT * 0.5),
        Vector2(-cap_width * 0.5, CAP_HEIGHT * 0.5)
    ])
    cap.color = Color("#48C878")
    parent.add_child(cap)

    var cap_collider := CollisionShape2D.new()
    cap_collider.position = Vector2(0, local_y)
    var cap_shape := RectangleShape2D.new()
    cap_shape.size = Vector2(cap_width, CAP_HEIGHT)
    cap_collider.shape = cap_shape
    parent.add_child(cap_collider)

func _build_hazard_area(center: Vector2, size: Vector2, cap_direction: float) -> void:
    # Area2D mengikuti gerak PipePair. Ini menjadi deteksi overlap
    # yang andal saat pipa bergerak menabrak Bird dari samping.
    var hazard_area := Area2D.new()
    hazard_area.position = center + Vector2(0, cap_direction * CAP_HEIGHT * 0.25)
    hazard_area.collision_layer = 0
    hazard_area.collision_mask = 1
    hazard_area.monitoring = true

    var hazard_shape := CollisionShape2D.new()
    var rect := RectangleShape2D.new()
    rect.size = Vector2(maxf(size.x, pipe_width + 14.0), size.y + CAP_HEIGHT * 0.5)
    hazard_shape.shape = rect
    hazard_area.add_child(hazard_shape)
    hazard_area.body_entered.connect(_on_hazard_body_entered)
    add_child(hazard_area)

func _on_hazard_body_entered(body: Node2D) -> void:
    if body is Bird and not has_crashed:
        has_crashed = true
        crashed.emit()

func _build_score_area() -> void:
    var score_area := Area2D.new()
    score_area.position = Vector2(pipe_width * 0.5 + 25.0, 0)
    score_area.collision_layer = 0
    score_area.collision_mask = 1
    score_area.monitoring = true

    var score_shape := CollisionShape2D.new()
    var rect := RectangleShape2D.new()
    rect.size = Vector2(18, 1000)
    score_shape.shape = rect
    score_area.add_child(score_shape)
    score_area.body_entered.connect(_on_score_body_entered)
    add_child(score_area)

func _on_score_body_entered(body: Node2D) -> void:
    if body is Bird and not has_scored:
        has_scored = true
        scored.emit()
