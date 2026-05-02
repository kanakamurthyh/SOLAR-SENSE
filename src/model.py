"""
model.py
========
Trains models, saves the best one, and exposes a predict function
for inference.

Flow:
    1. Load processed data (from disk if available, else reprocess)
    2. Train Linear Regression — baseline
    3. Train XGBoost — primary model
    4. Compare both, save the better one to models/
    5. predict(weather_df) — used by feasibility.py at inference time

Run directly to train:
    cd src/
    python model.py

Location: src/model.py
Imports:  src/utils/preprocessing.py, src/evaluate.py, src/config.py
Used by:  src/feasibility.py, app/app.py
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "utils"))

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from preprocessing import run_preprocessing, load_processed, split_data
from evaluate import evaluate_model, print_results
from config import (
    FEATURES, TARGET,
    XGBOOST_PARAMS,
    MODEL_PATH, METADATA_PATH,
)


# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────

def get_data():
    """
    Load processed data from disk if merged_clean.csv exists.
    Otherwise run the full preprocessing pipeline and save it.
    """
    df = load_processed()

    if df is None:
        # First time — process raw CSVs and save to disk
        X_train, X_test, y_train, y_test, capacity = run_preprocessing()
    else:
        # Already processed — just split
        capacity = float(df[TARGET].max())
        X_train, X_test, y_train, y_test = split_data(df)

    return X_train, X_test, y_train, y_test, capacity


# ─────────────────────────────────────────────────────────────
# 2. TRAIN
# ─────────────────────────────────────────────────────────────

def train_linear_regression(X_train, y_train):
    """
    Train a Linear Regression model — the class-taught baseline.

    Linear Regression finds weights w such that:
        y_pred = w1*x1 + w2*x2 + ... + wn*xn + b
    minimising the sum of squared errors between y_pred and y_true.

    Fast, interpretable, but cannot capture non-linear relationships
    — which is why it serves as our baseline to beat.
    """
    print("Training Linear Regression (baseline)...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  Done.\n")
    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost regressor — the primary model.

    XGBoost builds trees sequentially, each one correcting the
    errors of the previous. Hyperparameters are defined in config.py.

    early_stopping_rounds: stops training if test RMSE has not
    improved for 30 consecutive rounds — prevents overfitting
    without manually tuning n_estimators.
    """
    print("Training XGBoost...")
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    best_round = model.best_iteration
    print(f"  Best iteration: {best_round}")
    print("  Done.\n")
    return model


# ─────────────────────────────────────────────────────────────
# 3. SAVE
# ─────────────────────────────────────────────────────────────

def save_model(model, model_name, metrics, capacity):
    """
    Save the trained model and its metadata to models/ folder.

    metadata.json stores:
        features      — exact feature list and order used at training
        target        — target column name
        model_name    — which model was saved
        plant_capacity_kw — max AC power seen in training data
                            used to scale predictions to new plants
        metrics       — MAE, RMSE, nRMSE, R², skill_score

    feasibility.py loads both files at startup.
    """
    # Create models/ directory if it doesn't exist
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"  Saved model     → {MODEL_PATH}")

    metadata = {
        "features":           FEATURES,
        "target":             TARGET,
        "model_name":         model_name,
        "plant_capacity_kw":  capacity,
        "metrics":            metrics,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata  → {METADATA_PATH}\n")


# ─────────────────────────────────────────────────────────────
# 4. LOAD (used by feasibility.py at inference time)
# ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load the saved model and metadata from models/ folder.
    Called by feasibility.py before making any predictions.

    Returns:
        model     — trained sklearn/xgboost model
        metadata  — dict with features, capacity, metrics
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. "
            f"Run model.py first to train and save the model."
        )

    model    = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    print(f"  Loaded model: {metadata['model_name']}  "
          f"(R² = {metadata['metrics']['r2']:.4f})")
    return model, metadata


# ─────────────────────────────────────────────────────────────
# 5. PREDICT (called by feasibility.py)
# ─────────────────────────────────────────────────────────────

def predict(model, X):
    """
    Run inference on preprocessed feature DataFrame X.
    Returns numpy array of predicted kW values (clipped to >= 0).

    X must already have features in the exact order defined in
    config.FEATURES — prepare_inference_data() in preprocessing.py
    handles this automatically.
    """
    predictions = model.predict(X)
    return np.clip(predictions, 0, None)


# ─────────────────────────────────────────────────────────────
# MAIN — run this file directly to train
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  SolarSense — Model Training")
    print("=" * 50 + "\n")

    # ── Load data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, capacity = get_data()

    # ── Train Linear Regression ────────────────────────────────
    lr_model   = train_linear_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, capacity, "Linear Regression")
    print_results(lr_metrics)

    # ── Train XGBoost ──────────────────────────────────────────
    xgb_model   = train_xgboost(X_train, y_train, X_test, y_test)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, capacity, "XGBoost")
    print_results(xgb_metrics)

    # ── Compare and save the better model ─────────────────────
    # R² is the primary comparison metric —
    # higher R² means the model explains more variance in the data
    print("Comparing models...")
    if xgb_metrics["r2"] >= lr_metrics["r2"]:
        print(f"  XGBoost wins  "
              f"(R² {xgb_metrics['r2']:.4f} vs {lr_metrics['r2']:.4f})")
        save_model(xgb_model, "XGBoost", xgb_metrics, capacity)
    else:
        print(f"  Linear Regression wins  "
              f"(R² {lr_metrics['r2']:.4f} vs {xgb_metrics['r2']:.4f})")
        save_model(lr_model, "LinearRegression", lr_metrics, capacity)

    print("\nTraining complete. Run app/app.py to launch SolarSense.")


if __name__ == "__main__":
    main()