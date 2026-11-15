"""
Build the ML training set from REAL plant measurements.

For every plant in training/plants.py:
    1. load its metered hourly output (data/real_plants_hourly.csv)
    2. fetch Open-Meteo ERA5 reanalysis for the same years      -> "hindcast" rows
       fetch Open-Meteo archived NWP forecasts (2021+)           -> "forecast" rows
    3. fit any missing tilt from the plant's own POA sensor
    4. check clock alignment: shift each plant-year by the hour offset (-2..+2)
       that best matches measured POA to physics POA
    5. run physics with the plant's geometry, build ML features, attach target

Output: data/training_set.parquet with columns
    FEATURES + [target_pu, measured_poa_kwm2, phys_lossy_pu, plant, source, time]

Run from project root:
    python3 -m training.build_training_set
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from solarsense.config import REAL_PLANTS_PATH, TRAINING_SET_PATH
from solarsense.ml import make_features
from solarsense.physics import PVSystem, simulate
from solarsense.weather import fetch_openmeteo_archive, fetch_openmeteo_historical_forecast
from training.plants import PLANTS

HIST_FORECAST_START = pd.Timestamp("2021-03-24", tz="UTC")   # archive availability


def plant_frame(df: pd.DataFrame, pid: str) -> pd.DataFrame:
    p = df[df["LOCATION_ID"] == pid].copy()
    # plant rows are hour-start labelled UTC means -> interval centre
    p.index = pd.DatetimeIndex(pd.to_datetime(p["DATE_TIME"], utc=True) + pd.Timedelta(minutes=30), name="time")
    return p[~p.index.duplicated()]


def ac_limit(spec: dict, measured: pd.DataFrame) -> float:
    if spec["ac_kw"]:
        return float(spec["ac_kw"])
    return float(measured["AC_POWER"].quantile(0.999))


def fit_tilt(weather: pd.DataFrame, measured_poa: pd.Series, spec: dict) -> float:
    best = None
    for tilt in range(0, 41, 2):
        sim = simulate(weather, PVSystem(capacity_kwp=1, dc_ac_ratio=1, tilt=tilt, azimuth=spec["azimuth"]),
                       spec["lat"], spec["lon"], losses=False, clip=False)
        j = pd.concat([sim["poa_global"], measured_poa], axis=1, join="inner").dropna()
        j = j[j.iloc[:, 1] > 50]
        err = float(np.sqrt(np.mean((j.iloc[:, 0] - j.iloc[:, 1]) ** 2)))
        if best is None or err < best[1]:
            best = (tilt, err)
    return float(best[0])


def best_shift_hours(sim_poa: pd.Series, measured_poa: pd.Series) -> int:
    """Hour shift (applied to measured) that maximises correlation with physics POA."""
    scores = {}
    for h in range(-2, 3):
        shifted = measured_poa.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=h)
        j = pd.concat([sim_poa, shifted], axis=1, join="inner").dropna()
        if len(j) > 500:
            scores[h] = j.corr().iloc[0, 1]
    return max(scores, key=scores.get) if scores else 0


def build_rows(pid: str, spec: dict, measured: pd.DataFrame, weather: pd.DataFrame, source: str) -> pd.DataFrame:
    kwp = spec["kwp"]
    dc_ac = kwp / ac_limit(spec, measured)
    system = PVSystem(capacity_kwp=kwp, dc_ac_ratio=float(np.clip(dc_ac, 0.8, 2.0)),
                      tilt=spec["tilt"], azimuth=spec["azimuth"])
    sim_ideal = simulate(weather, system, spec["lat"], spec["lon"], losses=False, clip=True)
    sim_lossy = simulate(weather, system, spec["lat"], spec["lon"], losses=True, clip=True)
    feats = make_features(weather, sim_ideal, kwp, system.dc_ac_ratio)
    feats["phys_lossy_pu"] = sim_lossy["ac"] / kwp
    j = feats.join(measured[["AC_POWER", "IRRADIATION"]], how="inner")
    j["target_pu"] = j.pop("AC_POWER") / kwp
    j["measured_poa_kwm2"] = j.pop("IRRADIATION")
    j["plant"] = pid
    j["source"] = source
    return j[j["solar_elevation"] > 0]


def main():
    df = pd.read_csv(REAL_PLANTS_PATH)
    frames = []
    for pid, spec in PLANTS.items():
        spec = dict(spec)
        m = plant_frame(df, pid)
        y0, y1 = m.index.min().year, m.index.max().year
        print(f"[{pid}] {len(m):,} measured hours {m.index.min().date()}..{m.index.max().date()}", flush=True)

        era5 = fetch_openmeteo_archive(spec["lat"], spec["lon"], y0, y1)
        mpoa = m["IRRADIATION"] * 1000.0

        if spec["tilt"] is None:
            spec["tilt"] = fit_tilt(era5, mpoa, spec)
            print(f"  fitted tilt from POA sensor: {spec['tilt']:.0f}°")

        # clock alignment per year
        ref = simulate(era5, PVSystem(capacity_kwp=1, dc_ac_ratio=1, tilt=spec["tilt"], azimuth=spec["azimuth"]),
                       spec["lat"], spec["lon"], losses=False, clip=False)["poa_global"]
        parts = []
        for yr, part in m.groupby(m.index.year):
            h = best_shift_hours(ref, part["IRRADIATION"] * 1000.0)
            if h:
                print(f"  {yr}: shifting measured data {h:+d} h to match solar geometry")
                part = part.copy()
                part.index = part.index + pd.Timedelta(hours=h)
            parts.append(part)
        m = pd.concat(parts)
        m = m[~m.index.duplicated()]

        hind = build_rows(pid, spec, m, era5, "hindcast")
        frames.append(hind)
        print(f"  hindcast rows: {len(hind):,}")

        start = max(m.index.min(), HIST_FORECAST_START)
        if start < m.index.max():
            fc_parts = []
            for yr in range(start.year, m.index.max().year + 1):
                a = max(start, pd.Timestamp(f"{yr}-01-01", tz="UTC")).strftime("%Y-%m-%d")
                b = min(m.index.max(), pd.Timestamp(f"{yr}-12-31", tz="UTC")).strftime("%Y-%m-%d")
                fc_parts.append(fetch_openmeteo_historical_forecast(spec["lat"], spec["lon"], a, b))
            fc = pd.concat(fc_parts)
            fc = fc[~fc.index.duplicated()]
            fc.attrs = fc_parts[0].attrs
            fcrows = build_rows(pid, spec, m, fc, "forecast")
            frames.append(fcrows)
            print(f"  forecast rows: {len(fcrows):,}")

    out = pd.concat(frames)
    out.index.name = "time"
    out = out.reset_index()
    out.to_parquet(TRAINING_SET_PATH, index=False)
    print(f"\nSaved {len(out):,} rows -> {TRAINING_SET_PATH}")
    print(out.groupby(["plant", "source"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
