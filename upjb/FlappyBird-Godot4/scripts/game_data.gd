extends Node
## Data run disimpan di Autoload agar tetap tersedia saat scene berganti.

const TARGET_SCORE: int = 10

var score: int = 0
var best_score: int = 0
var won: bool = false

const SAVE_PATH := "user://savegame.cfg"

func _ready() -> void:
    _load_best_score()

func start_run() -> void:
    score = 0
    won = false

func add_point() -> int:
    score += 1
    if score > best_score:
        best_score = score
        _save_best_score()
    return score

func complete_run(is_won: bool) -> void:
    won = is_won

func _load_best_score() -> void:
    var config := ConfigFile.new()
    if config.load(SAVE_PATH) == OK:
        best_score = int(config.get_value("score", "best", 0))

func _save_best_score() -> void:
    var config := ConfigFile.new()
    config.set_value("score", "best", best_score)
    var error := config.save(SAVE_PATH)
    if error != OK:
        push_warning("Best score tidak dapat disimpan. Error: %s" % error)
