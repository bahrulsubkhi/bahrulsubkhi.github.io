extends SceneTree

var observed_score_signals := 0
var observed_crash_signals := 0

func _on_test_scored() -> void:
    observed_score_signals += 1

func _on_test_crashed() -> void:
    observed_crash_signals += 1

func _initialize() -> void:
    call_deferred("_run_tests")

func _run_tests() -> void:
    var main_scene: Node = load("res://scenes/main.tscn").instantiate()
    root.add_child(main_scene)
    current_scene = main_scene
    await process_frame
    if main_scene == null:
        push_error("Main scene is null")
        quit(1)
        return

    # Hindari menimpa best score pengguna saat tes otomatis.
    var game_data: Node = root.get_node("GameData")
    var previous_best_score := int(game_data.get("best_score"))
    game_data.set("best_score", 999999)

    var test_pipe := load("res://scenes/pipe_pair.tscn").instantiate() as PipePair
    test_pipe.position = Vector2(240, 330)
    test_pipe.scored.connect(_on_test_scored)
    test_pipe.crashed.connect(_on_test_crashed)
    root.add_child(test_pipe)

    var trigger_bird := load("res://scenes/bird.tscn").instantiate() as Bird
    trigger_bird.position = test_pipe.position + Vector2(62, 0)
    root.add_child(trigger_bird)
    for frame in range(3):
        await physics_frame
    if observed_score_signals != 1:
        push_error("Area2D score trigger did not fire exactly once")
        quit(1)
        return
    await physics_frame
    if observed_score_signals != 1:
        push_error("Score trigger fired more than once")
        quit(1)
        return
    if observed_crash_signals != 0:
        push_error("Bird in the pipe gap was incorrectly marked as crashed")
        quit(1)
        return
    trigger_bird.queue_free()
    await process_frame

    # Uji overlap dengan area bahaya, bukan hanya slide collision.
    var hazard_bird := load("res://scenes/bird.tscn").instantiate() as Bird
    hazard_bird.position = test_pipe.position + Vector2(0, -200)
    root.add_child(hazard_bird)
    for frame in range(3):
        await physics_frame
    if observed_crash_signals != 1:
        push_error("Pipe hazard did not emit crash on overlap")
        quit(1)
        return
    await physics_frame
    if observed_crash_signals != 1:
        push_error("Pipe hazard emitted crash more than once")
        quit(1)
        return
    print("PASS: upper pipe hazard detects Bird overlap exactly once")
    hazard_bird.queue_free()
    await process_frame

    var lower_pipe := load("res://scenes/pipe_pair.tscn").instantiate() as PipePair
    lower_pipe.position = Vector2(360, 330)
    lower_pipe.crashed.connect(_on_test_crashed)
    root.add_child(lower_pipe)

    var lower_hazard_bird := load("res://scenes/bird.tscn").instantiate() as Bird
    lower_hazard_bird.position = lower_pipe.position + Vector2(0, 200)
    root.add_child(lower_hazard_bird)
    for frame in range(3):
        await physics_frame
    if observed_crash_signals != 2:
        push_error("Lower pipe hazard did not detect Bird overlap")
        quit(1)
        return
    print("PASS: lower pipe hazard detects Bird overlap")
    lower_hazard_bird.queue_free()
    lower_pipe.queue_free()
    await process_frame

    var pipe_bodies: Array[AnimatableBody2D] = []
    for child in test_pipe.get_children():
        if child is AnimatableBody2D:
            pipe_bodies.append(child)
    if pipe_bodies.size() != 2:
        push_error("Expected exactly two pipe bodies")
        quit(1)
        return
    var upper_bottom := pipe_bodies[0].position.y + 380.0
    var lower_top := pipe_bodies[1].position.y - 380.0
    if absf(upper_bottom + 97.0) > 0.1 or absf(lower_top - 97.0) > 0.1:
        push_error("Pipe gap geometry is incorrect: %s / %s" % [upper_bottom, lower_top])
        quit(1)
        return
    test_pipe.queue_free()

    main_scene.call("_start_playing")
    for index in range(10):
        main_scene.call("_on_pipe_scored")
    await process_frame
    await process_frame

    var result_scene := current_scene
    if result_scene == null or result_scene.scene_file_path != "res://scenes/result.tscn":
        push_error("Win did not open the result scene")
        quit(1)
        return
    if game_data.get("score") != 10 or not game_data.get("won"):
        push_error("Win score/state is incorrect")
        quit(1)
        return
    print("PASS: pipe gap geometry")
    print("PASS: score reaches target and opens LEVEL CLEAR result")

    result_scene.call("_on_menu_pressed")
    await process_frame
    var menu_scene := current_scene
    if menu_scene == null or menu_scene.scene_file_path != "res://scenes/menu.tscn":
        push_error("Result -> Menu transition failed")
        quit(1)
        return

    menu_scene.call("_on_play_pressed")
    await process_frame
    main_scene = current_scene
    if main_scene == null or main_scene.scene_file_path != "res://scenes/main.tscn":
        push_error("Menu -> Gameplay transition failed")
        quit(1)
        return

    main_scene.call("_start_playing")
    main_scene.call("_pause_game")
    var test_bird: Bird = main_scene.get("bird")
    if int(main_scene.get("state")) != 2 or test_bird.can_move:
        push_error("Pause did not stop gameplay")
        quit(1)
        return
    main_scene.call("_resume_game")
    if int(main_scene.get("state")) != 1 or not test_bird.can_move:
        push_error("Resume did not restart gameplay")
        quit(1)
        return
    # Uji integrasi nyata: pipa spawn dari Main lalu overlap dengan Bird.
    test_bird.can_move = false
    main_scene.call("_spawn_pipe")
    var active_pipes: Array = main_scene.get("pipes")
    var hazard_pipe: PipePair = active_pipes[active_pipes.size() - 1]
    hazard_pipe.position = test_bird.position + Vector2(0, 200)
    for frame in range(3):
        await physics_frame
    # Main scene boleh sudah di-free saat transisi Result berlangsung.
    await process_frame
    await process_frame
    result_scene = current_scene
    if result_scene == null or result_scene.scene_file_path != "res://scenes/result.tscn":
        push_error("Lose did not open the result scene")
        quit(1)
        return
    var final_game_data: Node = root.get_node_or_null("GameData")
    if final_game_data == null or final_game_data.get("won"):
        push_error("Lose state was incorrectly marked as win")
        quit(1)
        return

    print("PASS: Result -> Menu -> Gameplay transitions")
    print("PASS: touching a spawned pipe ends the active run")
    print("PASS: pause and resume controls")
    print("PASS: crash opens GAME OVER result")
    print("ALL SMOKE TESTS PASSED")
    if is_instance_valid(final_game_data):
        final_game_data.set("best_score", previous_best_score)
    quit(0)
