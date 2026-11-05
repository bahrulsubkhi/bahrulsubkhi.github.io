extends CharacterBody2D
## Karakter pemain: input, gravitasi, velocity, rotasi dan collision.

signal crashed

@export var gravity: float = 980.0
@export var flap_impulse: float = -360.0
@export var max_fall_speed: float = 620.0
var can_move := true

func _physics_process(delta: float) -> void:
	if not can_move:
		return
	velocity.y = minf(velocity.y + gravity * delta, max_fall_speed)
	move_and_slide()
	rotation = lerp_angle(rotation, clampf(velocity.y / 700.0, -0.45, 0.85), delta * 5.0)
	for i in get_slide_collision_count():
		var collision := get_slide_collision(i)
		if collision:
			crashed.emit()

func flap() -> void:
	velocity.y = flap_impulse

func start() -> void:
	can_move = true

func stop() -> void:
	can_move = false
	velocity = Vector2.ZERO
