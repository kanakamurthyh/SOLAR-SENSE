"""
api.py
======
All external API calls in one place.
    get_forecast(lat, lon)   — 7-day hourly forecast from Open-Meteo
    get_historical(lat, lon) — 10-year monthly averages from NASA POWER

Location: src/api.py
Used by:  src/feasibility.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
import pandas as pd

from config import (
    OPENMETEO_FORECAST_URL,
    FORECAST_DAYS,
    NASA_POWER_URL,
    NASA_HISTORICAL_START,
    NASA_HISTORICAL_END,
)


# ─────────────────────────────────────────────────────────────
# OPEN-METEO FORECAST
# ─────────────────────────────────────────────────────────────

def get_forecast(lat, lon):
    """
    Fetch 7-day hourly weather forecast for any location.
    Free, no API key needed.
    """
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly": ",".join([
            "shortwave_radiation",
            "direct_radiation",
            "temperature_2m",
            "cloudcover",
            "windspeed_10m",
            "relativehumidity_2m",
        ]),
        "forecast_days": FORECAST_DAYS,
        "timezone":      "auto",
    }

    try:
        response = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Open-Meteo forecast API failed: {e}")

    data = response.json()["hourly"]
    df   = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])

    print(f"  Forecast fetched: {len(df)} hourly rows "
          f"({df['time'].min().date()} → {df['time'].max().date()})")

    return df


# ─────────────────────────────────────────────────────────────
# NASA POWER HISTORICAL
# ─────────────────────────────────────────────────────────────

def get_historical(lat, lon):
    """
    Fetch 10-year monthly average irradiance and temperature
    from NASA POWER for any location.

    NASA POWER can return data in two formats depending on the
    temporal resolution requested:
        - Yearly data: keys like "202301", "202302" (year+month)
        - Climatology: keys like "1", "2" ... "12" (just month number)

    We request climatology directly to avoid the parsing issue.

    Returns:
        dict with 'monthly_ghi' and 'monthly_temp' — both Series
        indexed 1–12 (Jan=1, Dec=12), or None if API fails.
    """
    # Use climatology endpoint — returns long-term monthly averages
    # directly as month numbers 1–12, no year prefix
    url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,T2M",
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "format":     "JSON",
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: NASA POWER API failed ({e}). "
              f"Feasibility analysis will be skipped.")
        return {"monthly_ghi": None, "monthly_temp": None}

    raw = response.json()["properties"]["parameter"]

    # Climatology keys are "1" through "13"
    # Key "13" is the annual average — we drop it
    ghi_raw  = raw["ALLSKY_SFC_SW_DWN"]
    temp_raw = raw["T2M"]

    # WITH this:

    MONTH_MAP = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
        # also handle numeric keys just in case
        "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6,
        "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12,
    }

    monthly_ghi  = pd.Series({
        MONTH_MAP[k]: float(v)
        for k, v in ghi_raw.items()
        if k in MONTH_MAP and float(v) != -999.0
    })
    monthly_temp = pd.Series({
        MONTH_MAP[k]: float(v)
        for k, v in temp_raw.items()
        if k in MONTH_MAP and float(v) != -999.0
    })

    monthly_ghi  = monthly_ghi.sort_index()
    monthly_temp = monthly_temp.sort_index()

    print(f"  Historical climatology fetched: 12 monthly averages")
    print(f"  GHI range: {monthly_ghi.min():.2f} – {monthly_ghi.max():.2f} kWh/m²/day")

    return {
        "monthly_ghi":  monthly_ghi,
        "monthly_temp": monthly_temp,
    }