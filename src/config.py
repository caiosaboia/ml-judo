import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "judo_atle_compt.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Elo settings
INITIAL_ELO = 1500
BASE_K = 32
ELO_SCALE = 400
