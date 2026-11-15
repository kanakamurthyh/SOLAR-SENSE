"""
Weather layer: every external weather source enters the package through this
module, and leaves it as a validated WeatherFrame.

WeatherFrame contract (enforced by `validate_weather`):
    index    tz-aware UTC DatetimeIndex, hourly, labelled at interval CENTRE
    ghi      global horizontal irradiance, W/m²
    dni      direct normal irradiance, W/m²
    dhi      diffuse horizontal irradiance, W/m²
    temp_air air temperature at 2 m, °C
    wind_speed  wind speed at 10 m, m/s
    optional: relative_humidity (%), cloud_cover (%)
    attrs    {"source": str, "latitude": float, "longitude": float, "elevation": float}

Keeping all unit and timezone conversion here is what prevents the W/m² vs
kW/m² and local-time vs UTC bugs of the v1 code.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import requests

from solarsense.config import (
    CACHE_DIR,
    FORECAST_DAYS,
    NASA_POWER_HOURLY_URL,
    OPENMETEO_ARCHIVE_URL,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_HISTORICAL_FORECAST_URL,
)

REQUIRED = ["ghi", "dni", "dhi", "temp_air", "wind_speed"]
OPTIONAL = ["relative_humidity", "cloud_cover"]

_OM_VARS = {
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "relative_humidity_2m": "relative_humidity",
    "cloud_cover": "cloud_cover",
}

WEATHER_CACHE = CACHE_DIR / "weather"


class WeatherError(RuntimeError):
    """Raised when a weather source cannot deliver valid data."""


# ─────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Check the WeatherFrame contract. Returns df unchanged or raises ValueError."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError("weather index must be a tz-aware DatetimeIndex")
    if str(df.index.tz) != "UTC":
        raise ValueError(f"weather index must be UTC, got {df.index.tz}")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"weather is missing columns {missing}")
    for col in ("ghi", "dni", "dhi"):
        mx = df[col].max()
        if mx > 1600:
            raise ValueError(f"{col} max {mx:.0f} > 1600: irradiance must be W/m²")
        if len(df) > 48 and 0 < mx < 2.0:
            raise ValueError(f"{col} max {mx:.3f}: looks like kW/m², must be W/m²")
        if df[col].min() < 0:
            raise ValueError(f"{col} has negative values")
    if not df["temp_air"].between(-70, 70).all():
        raise ValueError("temp_air outside -70..70 °C")
    if (df["wind_speed"] < 0).any() or (df["wind_speed"] > 75).any():
        raise ValueError("wind_speed outside 0..75 m/s (check km/h vs m/s)")
    return df


def _finalise(df: pd.DataFrame, source: str, lat: float, lon: float, elevation: float | None) -> pd.DataFrame:
    for col in ("ghi", "dni", "dhi"):
        df[col] = df[col].clip(lower=0).fillna(0.0)
    df["temp_air"] = df["temp_air"].interpolate(limit=3).ffill().bfill()
    df["wind_speed"] = df["wind_speed"].interpolate(limit=3).fillna(2.0).clip(lower=0)
    df = df[~df.index.duplicated()].sort_index()
    df.attrs.update(source=source, latitude=float(lat), longitude=float(lon),
                    elevation=float(elevation) if elevation is not None else np.nan)
    return validate_weather(df)


# ─────────────────────────────────────────────────────────────
# HTTP + CACHE HELPERS
# ─────────────────────────────────────────────────────────────

def _get_json(url: str, params: dict, timeout: int = 180, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                last = WeatherError(f"rate limited by {url}")
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and payload.get("error"):
                raise WeatherError(f"{url}: {payload.get('reason')}")
            return payload
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise WeatherError(f"request to {url} failed: {last}")


def _cache_path(kind: str, lat: float, lon: float, *parts) -> Path:
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    tail = "_".join(str(p) for p in parts)
    return WEATHER_CACHE / f"{kind}_{lat:.2f}_{lon:.2f}_{tail}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        meta = json.loads(df.attrs.get("meta", "{}")) if isinstance(df.attrs.get("meta"), str) else {}
        df.attrs.update(meta)
        return df
    except Exception:
        return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    out = df.copy()
    out.attrs = {"meta": json.dumps({k: v for k, v in df.attrs.items()})}
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp)
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────
# OPEN-METEO (archive = ERA5 reanalysis, forecast, historical forecast)
# ─────────────────────────────────────────────────────────────

def _openmeteo_frame(payload: dict, source: str, lat: float, lon: float) -> pd.DataFrame:
    h = payload["hourly"]
    df = pd.DataFrame({_OM_VARS[k]: h[k] for k in _OM_VARS if k in h})
    # Open-Meteo hourly radiation is the mean over the PRECEDING hour, so the
    # interval centre is 30 min before the label.
    idx = pd.to_datetime(h["time"], utc=True) - pd.Timedelta(minutes=30)
    df.index = pd.DatetimeIndex(idx, name="time")
    df = df.apply(pd.to_numeric, errors="coerce")
    return _finalise(df, source, lat, lon, payload.get("elevation"))


def _om_params(lat: float, lon: float) -> dict:
    return {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": ",".join(_OM_VARS),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }


def fetch_openmeteo_archive(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """ERA5-based hourly reanalysis for whole calendar years (cached)."""
    path = _cache_path("om_archive", lat, lon, start_year, end_year)
    cached = _read_cache(path)
    if cached is not None:
        return validate_weather(cached)
    params = _om_params(lat, lon) | {"start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31"}
    df = _openmeteo_frame(_get_json(OPENMETEO_ARCHIVE_URL, params), "open-meteo-era5", lat, lon)
    _write_cache(path, df)
    return df


def fetch_openmeteo_historical_forecast(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Archived short-lead NWP forecasts (available from 2021); used to train the forecast model."""
    path = _cache_path("om_histfc", lat, lon, start, end)
    cached = _read_cache(path)
    if cached is not None:
        return validate_weather(cached)
    params = _om_params(lat, lon) | {"start_date": start, "end_date": end}
    df = _openmeteo_frame(_get_json(OPENMETEO_HISTORICAL_FORECAST_URL, params), "open-meteo-historical-forecast", lat, lon)
    _write_cache(path, df)
    return df


def fetch_openmeteo_forecast(lat: float, lon: float, days: int = FORECAST_DAYS) -> pd.DataFrame:
    """Live hourly NWP forecast (not cached)."""
    params = _om_params(lat, lon) | {"forecast_days": days}
    return _openmeteo_frame(_get_json(OPENMETEO_FORECAST_URL, params, timeout=30), "open-meteo-forecast", lat, lon)


# ─────────────────────────────────────────────────────────────
# NASA POWER (fallback historical source)
# ─────────────────────────────────────────────────────────────

_NASA_PARAMS = "ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF,T2M,RH2M,WS10M"


def fetch_nasa_power(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """
    NASA POWER hourly (MERRA-2 + CERES). Requested explicitly in UTC: the API
    default is local solar time, which the v1 code silently consumed.
    Labels are hour-start, so the interval centre is +30 min.
    """
    path = _cache_path("nasa", lat, lon, start_year, end_year)
    cached = _read_cache(path)
    if cached is not None:
        return validate_weather(cached)
    params = {
        "parameters": _NASA_PARAMS, "community": "RE", "latitude": lat, "longitude": lon,
        "start": f"{start_year}0101", "end": f"{end_year}1231", "format": "JSON", "time-standard": "UTC",
    }
    payload = _get_json(NASA_POWER_HOURLY_URL, params, timeout=300)
    p = payload["properties"]["parameter"]
    raw = pd.DataFrame({k: pd.Series(v, dtype="float64") for k, v in p.items()})
    raw.index = pd.to_datetime(raw.index, format="%Y%m%d%H", utc=True) + pd.Timedelta(minutes=30)
    raw = raw.where(raw > -900)  # -999 fill values
    df = pd.DataFrame({
        "ghi": raw["ALLSKY_SFC_SW_DWN"],
        "dni": raw.get("ALLSKY_SFC_SW_DNI"),
        "dhi": raw.get("ALLSKY_SFC_SW_DIFF"),
        "temp_air": raw["T2M"],
        "wind_speed": raw["WS10M"],
        "relative_humidity": raw.get("RH2M"),
    }, index=raw.index.rename("time"))
    if df["dni"].isna().all() or df["dhi"].isna().all():
        df = fill_dni_dhi(df, lat, lon)
    elevation = payload.get("geometry", {}).get("coordinates", [None, None, None])[2]
    df = _finalise(df, "nasa-power", lat, lon, elevation)
    _write_cache(path, df)
    return df


def fill_dni_dhi(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Decompose GHI into DNI/DHI with the Erbs model when a source lacks them."""
    sp = pvlib.solarposition.get_solarposition(df.index, lat, lon)
    erbs = pvlib.irradiance.erbs(df["ghi"].fillna(0), sp["zenith"], df.index)
    df = df.copy()
    df["dni"] = erbs["dni"].fillna(0)
    df["dhi"] = erbs["dhi"].fillna(0)
    return df


# ─────────────────────────────────────────────────────────────
# HIGH-LEVEL: historical weather with fallback
# ─────────────────────────────────────────────────────────────

def get_history(lat: float, lon: float, start_year: int, end_year: int) -> pd.DataFrame:
    """Multi-year hourly history: Open-Meteo ERA5 first, NASA POWER as fallback."""
    errors = []
    for fetch in (fetch_openmeteo_archive, fetch_nasa_power):
        try:
            return fetch(lat, lon, start_year, end_year)
        except Exception as e:  # noqa: BLE001 - try the next source
            errors.append(f"{fetch.__name__}: {e}")
    raise WeatherError("all historical weather sources failed:\n  " + "\n  ".join(errors))


def complete_years(df: pd.DataFrame, min_coverage: float = 0.95) -> list[int]:
    """Calendar years with at least `min_coverage` of their hours present."""
    counts = df.groupby(df.index.year).size()
    hours = {y: (366 if pd.Timestamp(year=y, month=12, day=31).dayofyear == 366 else 365) * 24 for y in counts.index}
    return [int(y) for y, n in counts.items() if n >= min_coverage * hours[y]]
