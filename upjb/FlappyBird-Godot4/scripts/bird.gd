extends CharacterBody2D
class_name Bird
## Gerak vertikal, animasi sayap, dan deteksi collision karakter.

signal crashed

@export var gravity: float = 980.0
@export var flap_impulse: float = -360.0
@export var max_fall_speed: float = 620.0

var can_move := false
var has_crashed := false
var animation_time := 0.0

@onready var wing: Polygon2D = $Wing

func _process(delta: float) -> void:
    if can_move:
        animation_time += delta
        wing.rotation = sin(animation_time * 13.0) * 0.22
    else:
        wing.rotation = 0.0

func _physics_process(delta: float) -> void:
    if not can_move or has_crashed:
        return

    velocity.y = minf(velocity.y + gravity * delta, max_fall_speed)
    move_and_slide()
    rotation = lerp_angle(rotation,
        clampf(velocity.y / 700.0, -0.48, 0.82), delta * 5.5)

    if get_slide_collision_count() > 0:
        has_crashed = true
        can_move = false
        crashed.emit()

func flap() -> void:
    if can_move and not has_crashed:
        velocity.y = flap_impulse

func start() -> void:
    has_crashed = false
    can_move = true

func stop() -> void:
    can_move = false
    velocity = Vector2.ZERO
