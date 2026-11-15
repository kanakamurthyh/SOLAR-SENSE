"""
7-day probabilistic production forecast.

    Open-Meteo NWP forecast (UTC, W/m²)
      -> physics (user's plant, user's losses)            = deterministic shape
      -> x real-plant calibration k (forecast card)       = P50
      -> ML quantile half-widths x conformal scale
         x lead-day error growth                          = P10 / P90

Bands are calibrated on real plants the model never saw: 80 % of hourly
(and, separately, daily) outcomes fall inside P10-P90.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from solarsense.calibration import load_calibration
from solarsense.ml import QuantileModel, make_features
from solarsense.physics import PVSystem, simulate
from solarsense.weather import fetch_openmeteo_forecast


def local_timezone(lat: float, lon: float):
    """IST inside India's bounding box, otherwise a fixed offset from longitude."""
    if 6 <= lat <= 37.5 and 68 <= lon <= 97.5:
        return "Asia/Kolkata"
    return timezone(timedelta(hours=round(lon / 15)))


@dataclass
class ForecastResult:
    hourly: pd.DataFrame          # index UTC; local_time, ghi, cloud_cover, temp_air, physics_kw, p10_kw, p50_kw, p90_kw
    daily: pd.DataFrame           # date (local), p10_kwh, p50_kwh, p90_kwh, ghi_kwh_m2, cloud_cover_mean, peak_kw
    capacity_kwp: float
    ac_capacity_kw: float
    calibrated: bool
    k: float
    notes: list

    @property
    def total(self) -> dict:
        return {q: float(self.daily[f"{q}_kwh"].sum()) for q in ("p10", "p50", "p90")}


def _load_model() -> QuantileModel | None:
    try:
        return QuantileModel.load("forecast")
    except (FileNotFoundError, ValueError):
        return None


def run_forecast(lat: float, lon: float, system: PVSystem, weather: pd.DataFrame | None = None,
                 model: QuantileModel | None = None) -> ForecastResult:
    weather = fetch_openmeteo_forecast(lat, lon) if weather is None else weather
    cal = load_calibration("forecast")
    model = model if model is not None else _load_model()
    notes = []

    sim = simulate(weather, system, lat, lon)
    ac_cap = system.ac_capacity_kw
    p50 = (sim["ac"] * cal.k).clip(upper=ac_cap)

    lead = ((weather.index - weather.index[0]) / pd.Timedelta(days=1)).astype(int)
    lead_factor = np.array([cal.lead_factor(int(d)) for d in lead])

    if model is not None and cal.available:
        ideal = simulate(weather, system, lat, lon, losses=False, clip=True)
        q = model.predict(make_features(weather, ideal, system.capacity_kwp, system.dc_ac_ratio))
        lo_w = (q["p50"] - q["p10"]).clip(lower=0) * system.capacity_kwp * lead_factor
        hi_w = (q["p90"] - q["p50"]).clip(lower=0) * system.capacity_kwp * lead_factor
        s = cal.hourly_scale(sim["clearsky_index"])
        p10 = (p50 - s * lo_w).clip(lower=0)
        p90 = (p50 + s * hi_w).clip(upper=ac_cap)
    else:
        notes.append("ML uncertainty model not trained - showing physics only (run training).")
        lo_w = hi_w = pd.Series(0.0, index=weather.index)
        p10 = p90 = p50

    tz = local_timezone(lat, lon)
    hourly = pd.DataFrame({
        "local_time": weather.index.tz_convert(tz),
        "ghi": weather["ghi"],
        "cloud_cover": weather.get("cloud_cover", pd.Series(np.nan, index=weather.index)),
        "temp_air": weather["temp_air"],
        "clearsky_index": sim["clearsky_index"],
        "physics_kw": sim["ac"],
        "p10_kw": p10, "p50_kw": p50, "p90_kw": p90,
        "_lo_w": lo_w, "_hi_w": hi_w,
        "_csi_w": sim["clearsky_index"].fillna(0) * p50,
    }, index=weather.index)

    hourly["date"] = hourly["local_time"].dt.date
    g = hourly.groupby("date")
    daily = pd.DataFrame({
        "p50_kwh": g["p50_kw"].sum(),
        "lo_w": g["_lo_w"].sum(), "hi_w": g["_hi_w"].sum(),
        "csi": g["_csi_w"].sum() / g["p50_kw"].sum().where(g["p50_kw"].sum() > 0),
        "ghi_kwh_m2": g["ghi"].sum() / 1000.0,
        "cloud_cover_mean": g["cloud_cover"].mean(),
        "peak_kw": g["p50_kw"].max(),
        "hours": g.size(),
    })
    # partial first/last local days are dropped so totals are comparable
    daily = daily[daily["hours"] >= 20].copy()
    sd = cal.daily_scale(daily["csi"])
    daily["p10_kwh"] = (daily["p50_kwh"] - sd * daily["lo_w"]).clip(lower=0)
    daily["p90_kwh"] = daily["p50_kwh"] + sd * daily["hi_w"]
    daily = daily.rename(columns={"csi": "clearsky_index"}).drop(columns=["lo_w", "hi_w", "hours"]).reset_index()
    hourly = hourly.drop(columns=["_lo_w", "_hi_w", "_csi_w", "date"])

    return ForecastResult(hourly=hourly, daily=daily, capacity_kwp=system.capacity_kwp,
                          ac_capacity_kw=ac_cap, calibrated=cal.available, k=cal.k, notes=notes)
