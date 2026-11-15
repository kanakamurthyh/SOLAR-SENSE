"""
Golden test for the physics core: run solarsense.physics on Open-Meteo ERA5
weather for the 36 Indian cities and compare annual energy with PVGIS
(satellite irradiance + its own PV model) for the identical system
(1 kWp, 20° tilt, south, no extra system losses).

Writes reports/physics_vs_pvgis.csv and prints a summary.

Run from project root:
    python3 scripts/validate_physics_vs_pvgis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from solarsense.config import CACHE_DIR, INDIA_LOCATIONS_PATH, REPORTS_DIR  # noqa: E402
from solarsense.physics import PVSystem, simulate  # noqa: E402
from solarsense.weather import _finalise  # noqa: E402

OM_DIR = CACHE_DIR / "openmeteo_archive"
PVGIS_DIR = CACHE_DIR / "pvgis"


def load_om(city: str, lat: float, lon: float) -> pd.DataFrame:
    raw = pd.read_csv(OM_DIR / f"{city}_2020-01-01_2022-12-31.csv")
    df = pd.DataFrame({
        "ghi": raw["shortwave_radiation"], "dni": raw["direct_normal_irradiance"],
        "dhi": raw["diffuse_radiation"], "temp_air": raw["temperature_2m"],
        "wind_speed": raw["wind_speed_10m"], "relative_humidity": raw["relative_humidity_2m"],
        "cloud_cover": raw["cloud_cover"],
    })
    df.index = pd.DatetimeIndex(pd.to_datetime(raw["time"], utc=True) - pd.Timedelta(minutes=30), name="time")
    return _finalise(df, "open-meteo-era5", lat, lon, None)


def load_pvgis(city: str) -> pd.DataFrame:
    payload = json.loads((PVGIS_DIR / f"{city}_2020_2022_kw1_loss0_tilt20_az0.json").read_text())
    h = pd.DataFrame(payload["outputs"]["hourly"])
    h.index = pd.to_datetime(h["time"], format="%Y%m%d:%H%M", utc=True)
    return pd.DataFrame({"P_kw": pd.to_numeric(h["P"]) / 1000.0, "G_i": pd.to_numeric(h["G(i)"])})


def main():
    locs = pd.read_csv(INDIA_LOCATIONS_PATH)
    system = PVSystem(capacity_kwp=1.0, dc_ac_ratio=1.0, tilt=20.0, azimuth=180.0)
    rows = []
    for r in locs.itertuples(index=False):
        w = load_om(r.location_id, r.latitude, r.longitude)
        sim = simulate(w, system, r.latitude, r.longitude, losses=False, clip=False)
        pv = load_pvgis(r.location_id)
        years = 3.0
        ours_poa = sim["poa_global"].sum() / 1000 / years
        pvgis_poa = pv["G_i"].sum() / 1000 / years
        ours_e = sim["dc_gross"].sum() / years
        pvgis_e = pv["P_kw"].sum() / years
        # monthly shape agreement
        m_ours = sim["dc_gross"].groupby(sim.index.month).sum()
        m_pv = pv["P_kw"].groupby(pv.index.month).sum()
        rows.append({
            "city": r.city, "region": r.region,
            "poa_ours_kwh_m2": round(ours_poa), "poa_pvgis_kwh_m2": round(pvgis_poa),
            "yield_ours": round(ours_e), "yield_pvgis": round(pvgis_e),
            "yield_diff_pct": round((ours_e / pvgis_e - 1) * 100, 1),
            "monthly_corr": round(float(np.corrcoef(m_ours, m_pv)[0, 1]), 3),
            "max_month_diff_pct": round(float(((m_ours / m_pv - 1) * 100).abs().max()), 1),
        })
    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "physics_vs_pvgis.csv", index=False)
    print(df.to_string(index=False))
    d = df["yield_diff_pct"]
    print(f"\nAnnual yield diff vs PVGIS: mean {d.mean():+.1f}%  median {d.median():+.1f}%  "
          f"MAE {d.abs().mean():.1f}%  range {d.min():+.1f}..{d.max():+.1f}%")
    print(f"Monthly-shape correlation: median {df.monthly_corr.median():.3f}  min {df.monthly_corr.min():.3f}")


if __name__ == "__main__":
    main()
