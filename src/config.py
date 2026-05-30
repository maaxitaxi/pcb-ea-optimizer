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