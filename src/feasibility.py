"""
feasibility.py
==============
Connects weather data → model predictions → financial logic → verdict.

Two functions:
    run_forecast(lat, lon, capacity_kw, model, metadata)
        Uses Open-Meteo 7-day forecast to predict this week's output.

    run_feasibility(lat, lon, capacity_kw, cost_inr, tariff, model, metadata)
        Uses NASA POWER 10-year historical averages to estimate a typical
        year's output, then computes payback period, capacity factor,
        and returns a GO / MARGINAL / NO-GO verdict.

Both are called directly by app/app.py.

Location: src/feasibility.py
Imports:  src/api.py, src/utils/preprocessing.py, src/config.py
Used by:  app/app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "utils"))

import numpy as np
import pandas as pd
import math

from api import get_forecast, get_historical
from preprocessing import prepare_inference_data
from config import (
    FEATURES,
    CAPACITY_FACTOR_VIABLE,
    CAPACITY_FACTOR_MARGINAL,
    PAYBACK_YEARS_VIABLE,
    PAYBACK_YEARS_MARGINAL,
)

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# ─────────────────────────────────────────────────────────────
# SHARED HELPER — scale and predict
# ─────────────────────────────────────────────────────────────

def _predict_kwh(weather_df, model, training_capacity_kw, plant_capacity_kw):
    """
    Takes raw weather DataFrame (Open-Meteo format),
    runs feature engineering, predicts with the trained model,
    and scales output to the user's plant capacity.

    Returns a Series of predicted kWh values aligned to weather_df's
    non-nighttime rows.

    Why scaling:
        The model was trained on plants of a certain rated capacity.
        A user's new plant may be larger or smaller.
        We scale linearly: if training plant = 100 kW and user's = 500 kW,
        multiply all predictions by 5.
    """
    X = prepare_inference_data(weather_df)  # feature engineering + nighttime drop
    raw_predictions = model.predict(X)

    # Scale to user's plant capacity
    PHYSICAL_CAPACITY_KW = 29000  # actual rated capacity of training plants
    scale = plant_capacity_kw / PHYSICAL_CAPACITY_KW
    predictions = np.clip(raw_predictions * scale , 0, None)  # no negative kWh
    return predictions

# ─────────────────────────────────────────────────────────────
# OPERATIONAL FORECAST — 7-day
# ─────────────────────────────────────────────────────────────

def run_forecast(lat, lon, capacity_kw, model, metadata):
    """
    Fetch 7-day Open-Meteo forecast and predict hourly solar output.

    Returns a dictionary:
        'daily'        — DataFrame with columns: date, predicted_kwh
        'total_7day'   — float, total kWh over the 7-day window
        'hourly'       — DataFrame with columns: time, predicted_kw
                         (useful for plotting the daily curve)
        'avg_daily'    — float, average daily kWh across the 7 days
    """
    print(f"\nFetching 7-day forecast for ({lat}, {lon})...")
    forecast_df = get_forecast(lat, lon)

    training_capacity = metadata["plant_capacity_kw"]
    predictions = _predict_kwh(forecast_df, model, training_capacity, capacity_kw)

    # Attach predictions back to the daytime rows only
    # prepare_inference_data drops nighttime rows — we need to match indices
    daytime_mask = forecast_df["shortwave_radiation"] > 0
    hourly_df = forecast_df[daytime_mask].copy().reset_index(drop=True)
    hourly_df["predicted_kw"]  = predictions
    hourly_df["predicted_kwh"] = predictions * 1  # hourly data → kW = kWh per hour

    # Aggregate to daily totals
    hourly_df["date"] = hourly_df["time"].dt.date
    daily_df = (
        hourly_df.groupby("date")["predicted_kwh"]
        .sum()
        .reset_index()
        .rename(columns={"predicted_kwh": "predicted_kwh"})
    )

    total_7day = float(daily_df["predicted_kwh"].sum())
    avg_daily  = float(daily_df["predicted_kwh"].mean())

    print(f"  7-day total: {total_7day:.1f} kWh  |  Daily avg: {avg_daily:.1f} kWh")

    return {
        "daily":      daily_df,
        "hourly":     hourly_df[["time", "predicted_kw"]],
        "total_7day": total_7day,
        "avg_daily":  avg_daily,
    }


# ─────────────────────────────────────────────────────────────
# FEASIBILITY ANALYSIS — historical baseline
# ─────────────────────────────────────────────────────────────

def run_feasibility(lat, lon, capacity_kw, cost_inr, tariff, model, metadata):
    """
    Use NASA POWER 10-year monthly averages to estimate a typical
    year's output, then determine financial viability.

    How monthly estimation works:
        For each month (Jan–Dec) we build a synthetic 'typical day'
        using the historical average GHI and temperature for that month.
        We predict output for each daylight hour of that typical day,
        sum to get a typical day's kWh, then multiply by days in month
        to get the monthly estimate.

    Returns a dictionary:
        'monthly_kwh'       — dict {1: float, 2: float ... 12: float}
        'annual_kwh'        — float
        'annual_savings'    — float (₹)
        'payback_years'     — float
        'capacity_factor'   — float (%)
        'verdict'           — str: "VIABLE", "MARGINAL", or "NOT VIABLE"
        'verdict_emoji'     — str: "✅", "⚠️", or "❌"
        'reason'            — str: plain-English explanation of verdict
        'available'         — bool: False if NASA POWER call failed
    """
    print(f"\nFetching historical data for ({lat}, {lon})...")
    historical = get_historical(lat, lon)

    # If NASA POWER call failed, return gracefully
    if historical["monthly_ghi"] is None:
        return {"available": False}

    monthly_ghi  = historical["monthly_ghi"]   # Series index 1–12
    monthly_temp = historical["monthly_temp"]  # Series index 1–12
    training_capacity = metadata["plant_capacity_kw"]

    # ── Estimate output for each month ────────────────────────
    monthly_kwh = {}
    DAYLIGHT_HOURS = list(range(6, 19))   # 6am to 6pm — conservative daylight window

    for month_num in range(1, 13):
        ghi  = monthly_ghi[month_num]
        temp = monthly_temp[month_num]

        # Build a synthetic typical day for this month:
        # one row per daylight hour using the monthly average GHI and temp
        rows = []
        for h in DAYLIGHT_HOURS:
            daylight_hours_count = len(DAYLIGHT_HOURS)
            # Convert NASA POWER daily GHI (kWh/m²/day) to peak kW/m²
            # Daily GHI spread over daylight hours with bell curve
            peak_irr = (math.pi / 2) * ghi / daylight_hours_count

            hour_factor = max(0, 1 - ((h - 12) ** 2) / 25)
            hourly_irr  = min(peak_irr * hour_factor, 1.0)  # cap at 1.0 kW/m²

            rows.append({
                "time":                pd.Timestamp(f"2023-{month_num:02d}-15 {h:02d}:00"),
                "shortwave_radiation": hourly_irr,
                "temperature_2m":      temp,
                "relativehumidity_2m": 60,
                "cloudcover":          20,
                "windspeed_10m":       3,
                "direct_radiation":    hourly_irr * 0.7,
                "diffuse_radiation":   hourly_irr * 0.3,
            })

        typical_day_df = pd.DataFrame(rows)
        typical_day_df["time"] = pd.to_datetime(typical_day_df["time"])

        # Only keep rows where irradiance is positive
        typical_day_df = typical_day_df[typical_day_df["shortwave_radiation"] > 0]

        if typical_day_df.empty:
            monthly_kwh[month_num] = 0.0
            continue

        preds = _predict_kwh(
            typical_day_df, model, training_capacity, capacity_kw
        )

        # typical day kWh × days in month = monthly estimate
        typical_day_kwh = float(preds.sum())
        monthly_kwh[month_num] = typical_day_kwh * DAYS_IN_MONTH[month_num - 1]

    # ── Financial calculations ─────────────────────────────────
    annual_kwh     = sum(monthly_kwh.values())
    annual_savings = annual_kwh * tariff                          # ₹ saved per year
    payback_years  = cost_inr / annual_savings if annual_savings > 0 else 999

    # Capacity factor = actual annual output / theoretical max output
    # Theoretical max = capacity_kw running 24hrs × 365 days
    capacity_factor = (annual_kwh / (capacity_kw * 8760)) * 100  # %

    # ── Verdict ───────────────────────────────────────────────
   # ── Verdict ───────────────────────────────────────────────────
    financials_real = cost_inr > 100 and tariff > 0.5

    if financials_real:
        # Both CF and payback must qualify
        if capacity_factor >= CAPACITY_FACTOR_VIABLE and payback_years <= PAYBACK_YEARS_VIABLE:
            verdict       = "VIABLE"
            verdict_emoji = "✅"
            reason        = (f"Capacity factor {capacity_factor:.1f}% exceeds {CAPACITY_FACTOR_VIABLE}% "
                            f"and payback of {payback_years:.1f} yrs is within {PAYBACK_YEARS_VIABLE} yrs.")
        elif capacity_factor >= CAPACITY_FACTOR_MARGINAL and payback_years <= PAYBACK_YEARS_MARGINAL:
            verdict       = "MARGINAL"
            verdict_emoji = "⚠️"
            reason        = (f"Capacity factor {capacity_factor:.1f}% and payback {payback_years:.1f} yrs. "
                            f"Viable but review carefully.")
        else:
            verdict       = "NOT VIABLE"
            verdict_emoji = "❌"
            reason        = (f"Capacity factor {capacity_factor:.1f}% or payback {payback_years:.1f} yrs "
                            f"does not meet minimum thresholds.")
    else:
        # No financials — verdict based purely on capacity factor
        if capacity_factor >= CAPACITY_FACTOR_VIABLE:
            verdict       = "VIABLE"
            verdict_emoji = "✅"
            reason        = (f"Capacity factor {capacity_factor:.1f}% exceeds the "
                            f"{CAPACITY_FACTOR_VIABLE}% viability threshold.")
        elif capacity_factor >= CAPACITY_FACTOR_MARGINAL:
            verdict       = "MARGINAL"
            verdict_emoji = "⚠️"
            reason        = (f"Capacity factor {capacity_factor:.1f}% is between "
                            f"{CAPACITY_FACTOR_MARGINAL}% and {CAPACITY_FACTOR_VIABLE}%. "
                            f"Enable financial analysis for complete verdict.")
        else:
            verdict       = "NOT VIABLE"
            verdict_emoji = "❌"
            reason        = (f"Capacity factor {capacity_factor:.1f}% is below the "
                            f"minimum {CAPACITY_FACTOR_MARGINAL}% threshold.")

    print(f"  Annual output   : {annual_kwh:,.0f} kWh")
    print(f"  Annual savings  : ₹{annual_savings:,.0f}")
    print(f"  Payback period  : {payback_years:.1f} years")
    print(f"  Capacity factor : {capacity_factor:.1f}%")
    print(f"  Verdict         : {verdict} {verdict_emoji}")

    return {
        "monthly_kwh":      monthly_kwh,        # {1: float ... 12: float}
        "monthly_names":    MONTH_NAMES,         # ["Jan" ... "Dec"]
        "annual_kwh":       annual_kwh,
        "annual_savings":   annual_savings,
        "payback_years":    payback_years,
        "capacity_factor":  capacity_factor,
        "verdict":          verdict,
        "verdict_emoji":    verdict_emoji,
        "reason":           reason,
        "available":        True,
    }