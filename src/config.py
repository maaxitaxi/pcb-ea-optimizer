# config.py

MM_TO_NM = 1_000_000          # 1 mm = 1.000.000 Nanometer
NM_TO_MM = 1 / MM_TO_NM

# Platinengrenze
BOARD_WIDTH_MM  = 150
BOARD_HEIGHT_MM = 150
BOARD_W = BOARD_WIDTH_MM  * MM_TO_NM
BOARD_H = BOARD_HEIGHT_MM * MM_TO_NM

# Sicherheitsabstand um jedes Bauteil (Platz für Leiterbahnen/Vias)
COURTYARD_MARGIN = 2_500_000  # 2.5 mm


def set_board_size(width_nm: int, height_nm: int) -> None:
    """Überschreibt die Platinengröße, z.B. mit dem Umriss aus einer DSN-Datei."""
    global BOARD_W, BOARD_H, BOARD_WIDTH_MM, BOARD_HEIGHT_MM
    BOARD_W, BOARD_H = int(width_nm), int(height_nm)
    BOARD_WIDTH_MM, BOARD_HEIGHT_MM = BOARD_W / MM_TO_NM, BOARD_H / MM_TO_NM

VALID_ROTATIONS = [0, 90, 180, 270]

# EA-Hyperparameter (Optimiert für die neue GUI)
POP_SIZE        = 50      
TOURNAMENT_K    = 4       
CROSSOVER_RATE  = 0.8
ROT_MUTATE_P    = 0.25

# Mutation mit Abkühlung: am Anfang große Sprünge vieler Bauteile, später Feinjustierung weniger
MUTATION_RATE      = 0.3    # Anteil bewegter Bauteile pro Kind zu Beginn
MUTATION_MIN_PARTS = 1.5    # erwartete Anzahl bewegter Bauteile pro Kind am Ende
SIGMA_START_FRAC   = 0.25   # Schrittweite zu Beginn als Anteil der längeren Platinenkante
SIGMA_MIN_MM       = 0.5    # Schrittweite am Ende
SIGMA_MIN          = SIGMA_MIN_MM * MM_TO_NM
ANNEAL_GENS        = 150    # Zeitkonstante der Abkühlung (nach ~3× so vielen Generationen fast fertig)
MUTATION_TRIES     = 5      # Versuche, ein Bauteil überlappungsfrei zu verschieben, sonst zurücksetzen
SWAP_P             = 0.3    # Wahrscheinlichkeit pro Kind, zwei Bauteile die Plätze tauschen zu lassen
COMPACT_P          = 0.3    # Wahrscheinlichkeit pro Kind, Bauteile Richtung Gruppenmitte zu ziehen (schließt Lücken)

# Fitness-Gewichtung (muss in Summe 1.0 ergeben)
TRACE_WEIGHT    = 0.35   # Leitungslänge (netz-gewichtet)
OVERLAP_WEIGHT  = 0.45   # Bauteil-Überlappung
BBOX_WEIGHT     = 0.20   # Kompaktheit: Packungsdichte im Rechteck mit dem Seitenverhältnis der Platine

POWER_NET_KEYWORDS = ("VCC", "GND", "VIN", "3V3", "EN")
POWER_NET_WEIGHT   = 0.3
DATA_NET_WEIGHT    = 1.0

# Parallele Fitness-Bewertung (multiprocessing)
PARALLEL_WORKERS  = None    # None = CPU-Kerne - 1, 1 = aus
PARALLEL_MIN_WORK = 1_500   # erst ab (Pins × Populationsgröße) parallelisieren – darunter kostet der Overhead mehr