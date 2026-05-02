"""
config.py
=========
Single source of truth for all constants, file paths, and settings.
Every other file imports from here — nothing is hardcoded elsewhere.
Change something once here and it updates everywhere.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parent.parent 
#above line is defining the root directory of the project, which is one level up from the current file's directory
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

# Kaggle dataset CSV paths
PLANT1_GEN     = DATA_DIR / "Plant_1_Generation_Data.csv"
PLANT1_WEATHER = DATA_DIR / "Plant_1_Weather_Sensor_Data.csv"
PLANT2_GEN     = DATA_DIR / "Plant_2_Generation_Data.csv"
PLANT2_WEATHER = DATA_DIR / "Plant_2_Weather_Sensor_Data.csv"

# Saved model outputs
MODEL_PATH    = MODELS_DIR / "xgboost_model.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"


# ─────────────────────────────────────────────────────────────
# FEATURES & TARGET
# ─────────────────────────────────────────────────────────────

# Exact list of features the model trains on and predicts from.
# Order matters — inference data must match this order exactly.
FEATURES = [
    "IRRADIATION",           # raw — strongest predictor
    "AMBIENT_TEMPERATURE",   # raw
    "MODULE_TEMPERATURE",    # raw (or estimated at inference)
    "temp_delta",            # derived — panel temp minus air temp
    "irradiation_sq",        # derived — captures non-linear curve
    "irr_x_temp",            # derived — interaction term
    "hour_sin",              # time — cyclic hour encoding
    "hour_cos",
    "month_sin",             # time — cyclic month encoding
    "month_cos",
    "day_of_year",           # time — seasonal position in year
    "PLANT_ID",              # which plant (0 = unknown at inference)
]

TARGET = "AC_POWER"


# ─────────────────────────────────────────────────────────────
# PREPROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────

TRAIN_SPLIT_RATIO  = 0.8    # 80% train, 20% test (by date)
OUTLIER_CLIP_LOW   = 0.01   # remove bottom 1% as outliers
OUTLIER_CLIP_HIGH  = 0.99   # remove top 1% as outliers


# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

XGBOOST_PARAMS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "max_depth":         6,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":            -1,
    "early_stopping_rounds": 30,
    "eval_metric":       "rmse",
}
 

# ─────────────────────────────────────────────────────────────
# API SETTINGS
# ─────────────────────────────────────────────────────────────

# Open-Meteo — free, no API key needed
OPENMETEO_FORECAST_URL   = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_HISTORICAL_URL = "https://archive.open-meteo.com/v1/archive"
FORECAST_DAYS            = 7

# NASA POWER — free, no API key needed
NASA_POWER_URL           = "https://power.larc.nasa.gov/api/temporal/monthly/point"
NASA_HISTORICAL_START    = 2013
NASA_HISTORICAL_END      = 2023


# ─────────────────────────────────────────────────────────────
# FEASIBILITY THRESHOLDS
# ─────────────────────────────────────────────────────────────

CAPACITY_FACTOR_VIABLE   = 18.0   # % — minimum CF for a GO verdict (usually between 15–20% depending on location and economics)
CAPACITY_FACTOR_MARGINAL = 14.0   # % — minimum CF for MARGINAL verdict (usually between 10–15%)
PAYBACK_YEARS_VIABLE     = 10     # years — maximum payback for GO 
PAYBACK_YEARS_MARGINAL   = 15     # years — maximum payback for MARGINAL (usually between 10–20 years depending on economics)