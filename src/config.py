# config.py

MM_TO_NM = 1_000_000          # 1 mm = 1.000.000 Nanometer
NM_TO_MM = 1 / MM_TO_NM

# Platinengrenze
BOARD_WIDTH_MM  = 150
BOARD_HEIGHT_MM = 150
BOARD_W = BOARD_WIDTH_MM  * MM_TO_NM
BOARD_H = BOARD_HEIGHT_MM * MM_TO_NM

VALID_ROTATIONS = [0, 90, 180, 270]

# EA-Hyperparameter (Optimiert für die neue GUI)
POP_SIZE        = 50      
TOURNAMENT_K    = 4       
MUTATION_RATE   = 0.3     # Leicht erhöht für mehr Dynamik
CROSSOVER_RATE  = 0.8     
SIGMA_XY_MM     = 8.0     
SIGMA_XY        = SIGMA_XY_MM * MM_TO_NM
ROT_MUTATE_P    = 0.25

# Fitness-Gewichtung (muss in Summe 1.0 ergeben)
TRACE_WEIGHT    = 0.35   # Leitungslänge (netz-gewichtet, siehe unten)
OVERLAP_WEIGHT  = 0.45   # Bauteil-Überlappung
BBOX_WEIGHT     = 0.20   # Free-Space-Minimization (kompakte Bounding Box ums Layout)

# Netz-Gewichtung für die Leitungslängen-Berechnung:
# Power-Nets (Spannungsversorgung) werden geringer gewichtet als Daten-/Signal-Nets,
# da kurze Versorgungsleitungen weniger kritisch sind als Signalintegrität bei Daten-Leitungen.
POWER_NET_KEYWORDS = ("VCC", "GND", "VIN", "3V3", "EN")
POWER_NET_WEIGHT   = 0.3
DATA_NET_WEIGHT    = 1.0