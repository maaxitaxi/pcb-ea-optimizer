# config.py

MM_TO_NM = 1_000_000          # 1 mm = 1.000.000 Nanometer
NM_TO_MM = 1 / MM_TO_NM

# Platinengrenze
BOARD_WIDTH_MM  = 150
BOARD_HEIGHT_MM = 150
BOARD_W = BOARD_WIDTH_MM  * MM_TO_NM
BOARD_H = BOARD_HEIGHT_MM * MM_TO_NM

VALID_ROTATIONS = [0, 90, 180, 270]

# ---------------------------------------------------------------------------
# EA-Hyperparameter
# ---------------------------------------------------------------------------
POP_SIZE        = 50
TOURNAMENT_K    = 4
CROSSOVER_RATE  = 0.8
ROT_MUTATE_P    = 0.25

# Statische Fallback-Werte, falls keine dynamische Schedule verwendet wird
MUTATION_RATE   = 0.3
SIGMA_XY_MM     = 8.0
SIGMA_XY        = SIGMA_XY_MM * MM_TO_NM

# ---------------------------------------------------------------------------
# Neue Analyse-/Experiment-Parameter
# ---------------------------------------------------------------------------
# Idee aus dem Treffen: am Anfang den Suchraum möglichst groß machen.
# Deshalb startet der Lauf mit hoher Mutation / großer Schrittweite und einer
# weniger harten Overlap-Strafe. Richtung Ende wird stärker ausgenutzt und
# Overlap stärker bestraft.
USE_DYNAMIC_SCHEDULE = True
SCHEDULE_GENERATIONS = 250

# Initialpopulation: False bedeutet, dass auch überlappende Startlayouts erlaubt
# sind. Dadurch wird der Suchraum am Anfang weniger eingeschränkt.
INITIAL_AVOID_OVERLAP = False

# Dynamische Gewichtung der Fitness: trace_weight + overlap_weight = 1.0
START_OVERLAP_WEIGHT = 0.45
END_OVERLAP_WEIGHT   = 0.90

# Dynamische Mutationsparameter: große Exploration am Anfang, kleinere Schritte
# gegen Ende.
START_MUTATION_RATE = 0.65
END_MUTATION_RATE   = 0.18
START_SIGMA_XY_MM   = 25.0
END_SIGMA_XY_MM     = 4.0

# CSV/Plot-Ausgaben
DEFAULT_HISTORY_CSV  = "run_history.csv"
DEFAULT_HISTORY_PLOT = "run_history.png"