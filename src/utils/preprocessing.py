"""
preprocessing.py
================
Handles everything between raw CSV files and model-ready data.
    1. Load & merge Kaggle plant CSVs
    2. Clean (drop bad rows, outliers, nighttime)
    3. Engineer features from raw columns + timestamp
    4. Split into train / test by date (never random)
    5. prepare_inference_data() — same pipeline for Open-Meteo forecast data

Location: src/utils/preprocessing.py
Used by:  src/model.py, src/feasibility.py
"""

import sys
from pathlib import Path

# Go up one level from src/utils/ → src/ to reach config.py
SRC_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
from config import (
    PLANT1_GEN, PLANT1_WEATHER,
    PLANT2_GEN, PLANT2_WEATHER,
    FEATURES, TARGET,
    TRAIN_SPLIT_RATIO,
    OUTLIER_CLIP_LOW, OUTLIER_CLIP_HIGH,
    DATA_DIR,
)

PROCESSED_PATH = DATA_DIR / "merged_clean.csv"


# ─────────────────────────────────────────────────────────────
# 1. LOAD & MERGE
# ─────────────────────────────────────────────────────────────

def _load_plant(gen_path, weather_path, plant_id):
    gen     = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)

    gen["DATE_TIME"]     = pd.to_datetime(gen["DATE_TIME"],     dayfirst=True)
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], dayfirst=True)

    # Aggregate all inverters per timestamp into one row
    gen = gen.groupby("DATE_TIME", as_index=False).agg({
        "AC_POWER":    "sum",
        "DC_POWER":    "sum",
        "DAILY_YIELD": "sum",
    })

    df = pd.merge(gen, weather, on="DATE_TIME", how="inner")
    df["PLANT_ID"] = plant_id

    print(f"  Plant {plant_id}: {len(df)} merged rows after inverter aggregation")
    return df

def load_raw_data():
    print("Loading plant data...")
    # loading two plant data by calling _load_plant function 
    plant1 = _load_plant(PLANT1_GEN, PLANT1_WEATHER, plant_id=1)
    plant2 = _load_plant(PLANT2_GEN, PLANT2_WEATHER, plant_id=2)

    df = pd.concat([plant1, plant2], ignore_index=True) # concat adds the two dataframes together 
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    print(f"  Combined total: {len(df):,} rows\n")
    return df

# ─────────────────────────────────────────────────────────────
# 2. CLEAN
# ─────────────────────────────────────────────────────────────

def clean(df):
    print("Cleaning data...")
    before = len(df)

    df = df.dropna(subset=["AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE"])
    df = df[df["AC_POWER"] >= 0]
    df = df[df["IRRADIATION"] > 0]

    for col in ["AC_POWER", "IRRADIATION"]:
        low  = df[col].quantile(OUTLIER_CLIP_LOW)
        high = df[col].quantile(OUTLIER_CLIP_HIGH)
        df   = df[(df[col] >= low) & (df[col] <= high)]

    after = len(df)
    print(f"  Removed {before - after:,} rows "
          f"({(before - after) / before * 100:.1f}%)")
    print(f"  Remaining: {after:,} rows\n")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df):
    """
    Called during BOTH training and inference — guarantees
    features are always computed identically in both cases.
    """
    df = df.copy()

    hour        = df["DATE_TIME"].dt.hour
    month       = df["DATE_TIME"].dt.month
    day_of_year = df["DATE_TIME"].dt.dayofyear

    df["hour_sin"]    = np.sin(2 * np.pi * hour  / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * hour  / 24)
    df["month_sin"]   = np.sin(2 * np.pi * month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * month / 12)
    df["day_of_year"] = day_of_year

    if "MODULE_TEMPERATURE" not in df.columns:
        df["MODULE_TEMPERATURE"] = (
            df["AMBIENT_TEMPERATURE"] + (df["IRRADIATION"] * 0.0342)
        )

    df["temp_delta"]     = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
    df["irradiation_sq"] = df["IRRADIATION"] ** 2
    df["irr_x_temp"]     = df["IRRADIATION"] * df["AMBIENT_TEMPERATURE"]

    if "PLANT_ID" not in df.columns:
        df["PLANT_ID"] = 0

    return df


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

def split_data(df):
    df = df.sort_values("DATE_TIME").reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_SPLIT_RATIO)

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    print(f"Train: {len(train):,} rows  "
          f"({train['DATE_TIME'].min().date()} -> "
          f"{train['DATE_TIME'].max().date()})")
    print(f"Test:  {len(test):,} rows  "
          f"({test['DATE_TIME'].min().date()} -> "
          f"{test['DATE_TIME'].max().date()})\n")

    X_train = train[FEATURES]
    X_test  = test[FEATURES]
    y_train = train[TARGET]
    y_test  = test[TARGET]

    return X_train, X_test, y_train, y_test
#Why that matters
#This is a time-series split, not a random split.
#It keeps earlier data in training and later data in testing.
#That avoids data leakage from future timestamps into training.

# ─────────────────────────────────────────────────────────────
# 5. INFERENCE PIPELINE
# ─────────────────────────────────────────────────────────────

def prepare_inference_data(weather_df):
    df = weather_df.copy()

    df = df.rename(columns={
        "shortwave_radiation": "IRRADIATION",
        "temperature_2m":      "AMBIENT_TEMPERATURE",
        "time":                "DATE_TIME",
    })

    df = df[df["IRRADIATION"] > 0].copy()
    df = engineer_features(df)

    return df[FEATURES]


# ─────────────────────────────────────────────────────────────
# 6. LOAD PROCESSED (skip reprocessing if file exists)
# ─────────────────────────────────────────────────────────────

def load_processed():
    """
    Load merged_clean.csv from disk if it already exists.
    Returns the DataFrame, or None if file not found.

    model.py uses this pattern:
        df = load_processed()
        if df is None:
            run_preprocessing()   # first time — processes and saves
        else:
            split_data(df)        # already processed — load and split
    """
    if PROCESSED_PATH.exists():
        print(f"  Found processed file — loading {PROCESSED_PATH}")
        df = pd.read_csv(PROCESSED_PATH)
        df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
        print(f"  Loaded {len(df):,} rows\n")
        return df

    print("  No processed file found — running full preprocessing...\n")
    return None


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — called by src/model.py
# ─────────────────────────────────────────────────────────────

def run_preprocessing():
    """
    Full pipeline: load -> clean -> engineer -> save to disk -> split.
    Saves merged_clean.csv so subsequent runs skip reprocessing.
    """
    df = load_raw_data()
    df = clean(df)
    df = engineer_features(df)

    # Save to disk BEFORE return — otherwise this line never runs
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"  Saved processed data -> {PROCESSED_PATH}\n")

    capacity = float(df[TARGET].quantile(0.85))   # 19,773 kW
    X_train, X_test, y_train, y_test = split_data(df)

    return X_train, X_test, y_train, y_test, capacity