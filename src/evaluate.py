"""
evaluate.py
===========
Computes all evaluation metrics and generates plots after training.

Two functions used by model.py:
    evaluate_model()   — computes MAE, RMSE, nRMSE, R², skill score
    print_results()    — prints a clean summary to terminal

Two functions used by app.py (optional visual diagnostics):
    plot_predicted_vs_actual()  — line chart of predicted vs actual
    plot_feature_importance()   — bar chart of XGBoost feature weights

Location: src/evaluate.py
Imports:  src/config.py
Used by:  src/model.py, app/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import FEATURES


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, training_capacity, model_name="Model"):
    """
    Compute all evaluation metrics on the test set.

    Metrics explained:

    MAE (Mean Absolute Error)
        Average absolute difference between predicted and actual kW.
        Easy to interpret — "on average we are off by X kW."

    RMSE (Root Mean Squared Error)
        Like MAE but penalises large errors more heavily.
        Useful for catching occasional bad predictions.

    nRMSE (Normalised RMSE)
        RMSE expressed as a % of the plant's rated capacity.
        Industry standard — allows comparison across plants of
        different sizes. Target: below 10%.

    R² (Coefficient of Determination)
        How much of the variance in output the model explains.
        1.0 = perfect, 0.0 = no better than predicting the mean.
        Target: above 0.93 for XGBoost.

    Skill Score
        How much better is our model vs a naive persistence baseline?
        Persistence = "next hour = current hour" (dumbest forecast).
        Skill score > 0 means we beat the baseline.
        Formula: (1 - RMSE_model / RMSE_persistence) × 100
        A skill score of 40% means our RMSE is 40% lower than persistence.

    Returns a dict with all metrics — saved to metadata.json by model.py.
    """
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # nRMSE — normalise by training plant's rated capacity
    nrmse = (rmse / training_capacity) * 100

    # Skill score vs persistence baseline
    # Persistence: shift y_test by 1 step (previous value as prediction)
    y_test_arr   = np.array(y_test)
    y_persistence = y_test_arr[:-1]        # previous hour values
    y_actual_next = y_test_arr[1:]         # next hour actual values

    rmse_persistence = np.sqrt(mean_squared_error(y_actual_next, y_persistence))
    skill_score      = (1 - rmse / rmse_persistence) * 100 if rmse_persistence > 0 else 0.0

    return {
        "model_name":   model_name,
        "mae":          round(float(mae),   4),
        "rmse":         round(float(rmse),  4),
        "nrmse_pct":    round(float(nrmse), 4),
        "r2":           round(float(r2),    4),
        "skill_score":  round(float(skill_score), 2),
    }


def print_results(metrics):
    """Print a clean metrics summary to terminal."""
    name = metrics["model_name"]
    print(f"{'─' * 42}")
    print(f"  Results — {name}")
    print(f"{'─' * 42}")
    print(f"  MAE          : {metrics['mae']:.4f} kW")
    print(f"  RMSE         : {metrics['rmse']:.4f} kW")
    print(f"  nRMSE        : {metrics['nrmse_pct']:.2f}%  (target < 10%)")
    print(f"  R²           : {metrics['r2']:.4f}  (target > 0.93)")
    print(f"  Skill score  : {metrics['skill_score']:.1f}%  (vs persistence baseline)")
    print(f"{'─' * 42}\n")


# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

def plot_predicted_vs_actual(model, X_test, y_test, save_path=None, n_days=4):
    """
    Plot predicted vs actual AC power over a sample of n_days.

    Why n_days and not all test data:
        The test set may have thousands of rows — plotting all of them
        produces an unreadable dense line. A 4-day window clearly shows
        whether the model tracks the daily wave pattern correctly.

    Returns:
        matplotlib Figure object (used directly by Streamlit in app.py)
        Also saves to save_path if provided.
    """
    y_pred = np.clip(model.predict(X_test), 0, None)

    # Take first n_days × 96 rows (15-min data = 96 rows per day)
    # If hourly data, n_days × 24
    n_rows = min(n_days * 96, len(y_test))
    actual = np.array(y_test)[:n_rows]
    pred   = y_pred[:n_rows]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(actual, label="Actual",    color="#1D9E75", linewidth=1.5)
    ax.plot(pred,   label="Predicted", color="#378ADD", linewidth=1.5,
            linestyle="--")
    ax.set_title(f"Predicted vs Actual AC Power — {n_days}-day sample",
                 fontsize=13)
    ax.set_ylabel("AC Power (kW)")
    ax.set_xlabel("Time steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_feature_importance(model, save_path=None):
    """
    Bar chart of XGBoost feature importance scores.

    Feature importance shows how much each feature contributed
    to reducing prediction error across all trees.
    The dominant feature should be IRRADIATION — if it isn't,
    something went wrong in preprocessing.

    Only works for XGBoost — LinearRegression does not have
    feature_importances_. Checks model type before plotting.

    Returns:
        matplotlib Figure object, or None if model is not XGBoost.
    """
    if not hasattr(model, "feature_importances_"):
        print("  Skipping feature importance — not an XGBoost model.")
        return None

    importances = model.feature_importances_
    indices     = np.argsort(importances)
    sorted_features = [FEATURES[i] for i in indices]
    sorted_importance = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features, sorted_importance, color="#378ADD")
    ax.set_title("Feature importance (XGBoost)", fontsize=13)
    ax.set_xlabel("Importance score")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_monthly_output(monthly_kwh, monthly_names, capacity_kw, save_path=None):
    """
    Bar chart of estimated monthly energy output.
    Used in the feasibility tab of app.py.

    Returns matplotlib Figure object.
    """
    values = [monthly_kwh[m] for m in range(1, 13)]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(monthly_names, values, color="#1D9E75", edgecolor="white")
    ax.set_title(f"Estimated monthly output — {capacity_kw} kW plant", fontsize=13)
    ax.set_ylabel("kWh / month")
    ax.grid(True, axis="y", alpha=0.3)

    # Label each bar with its value
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=8, color="#444"
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_forecast(daily_df, save_path=None):
    """
    Bar chart of 7-day predicted output.
    Used in the forecast tab of app.py.

    daily_df must have columns: date, predicted_kwh
    Returns matplotlib Figure object.
    """
    labels = [str(d) for d in daily_df["date"]]
    values = daily_df["predicted_kwh"].tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color="#378ADD", edgecolor="white")
    ax.set_title("7-day solar output forecast", fontsize=13)
    ax.set_ylabel("Predicted kWh")
    plt.xticks(rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=9, color="#444"
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    return fig