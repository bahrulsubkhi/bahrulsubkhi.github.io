extends Node2D
## Rintangan bergerak secara sinkron dengan physics frame.
## AnimatableBody2D cocok untuk obstacle yang digerakkan manual.

signal scored

@export var speed: float = 185.0
@export var gap_size: float = 178.0
@export var pipe_width: float = 72.0

var has_scored := false

func _ready() -> void:
	# Posisi Node2D induk sudah menjadi titik tengah celah.
	_build_pipe(Vector2(0, -gap_size * 0.5 - 300.0), Vector2(pipe_width, 600.0))
	_build_pipe(Vector2(0, gap_size * 0.5), Vector2(pipe_width, 600.0))

	var score_area := Area2D.new()
	score_area.position = Vector2(pipe_width + 16.0, 0)
	score_area.collision_layer = 0
	score_area.collision_mask = 1
	var score_shape := CollisionShape2D.new()
	var rect := RectangleShape2D.new()
	rect.size = Vector2(30, 720)
	score_shape.shape = rect
	score_area.add_child(score_shape)
	score_area.body_entered.connect(_on_score_body_entered)
	add_child(score_area)

func _physics_process(delta: float) -> void:
	position.x -= speed * delta
	if position.x < -120.0:
		queue_free()

func _build_pipe(center: Vector2, size: Vector2) -> void:
	var body := AnimatableBody2D.new()
	body.position = center
	body.collision_layer = 2
	body.collision_mask = 0
	body.sync_to_physics = true

	var visual := Polygon2D.new()
	var half_w := size.x * 0.5
	var half_h := size.y * 0.5
	visual.polygon = PackedVector2Array([
		Vector2(-half_w, -half_h), Vector2(half_w, -half_h),
		Vector2(half_w, half_h), Vector2(-half_w, half_h)
	])
	visual.color = Color("#36b85e")
	body.add_child(visual)

	var collider := CollisionShape2D.new()
	var shape := RectangleShape2D.new()
	shape.size = size
	collider.shape = shape
	body.add_child(collider)
	add_child(body)

func _on_score_body_entered(body: Node2D) -> void:
	if body is CharacterBody2D and not has_scored:
		has_scored = true
		scored.emit()
