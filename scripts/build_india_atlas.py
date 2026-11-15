"""
India Solar Atlas: calibrated P50 yield, CUF, seasonality and optimal tilt
for the 36 reference cities, from cached Open-Meteo ERA5 weather (2020-2022).

Writes data/india_atlas.csv (used by the app's Atlas tab).

Run from project root (after scripts/fetch_openmeteo_india_archive.py and
training, so the real-plant calibration is applied):
    python3 scripts/build_india_atlas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.validate_physics_vs_pvgis import load_om  # noqa: E402
from solarsense.calibration import load_calibration  # noqa: E402
from solarsense.config import ATLAS_PATH, INDIA_LOCATIONS_PATH  # noqa: E402
from solarsense.physics import PVSystem, optimise_orientation, simulate  # noqa: E402

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def main():
    cal = load_calibration("hindcast")
    rows = []
    for r in pd.read_csv(INDIA_LOCATIONS_PATH).itertuples(index=False):
        w = load_om(r.location_id, r.latitude, r.longitude)
        system = PVSystem(capacity_kwp=1.0)
        sim = simulate(w, system, r.latitude, r.longitude)
        years = len(w) / 8766
        ac = sim["ac"] * cal.k
        sy = ac.sum() / years
        monthly = ac.groupby(ac.index.month).sum() / years
        dry = monthly.drop(index=[6, 7, 8, 9]).mean()
        opt = optimise_orientation(w, system, r.latitude, r.longitude, tilts=range(0, 41, 4)).iloc[0]
        rows.append({
            "location_id": r.location_id, "city": r.city, "state": r.state, "region": r.region,
            "latitude": r.latitude, "longitude": r.longitude,
            "ghi_kwh_m2": round(w["ghi"].sum() / 1000 / years),
            "specific_yield_p50": round(sy),
            "cuf_ac_pct": round(sy / (8760 / system.dc_ac_ratio) * 100, 2),
            "best_month": MONTHS[int(monthly.idxmax()) - 1],
            "worst_month": MONTHS[int(monthly.idxmin()) - 1],
            "monsoon_dip_pct": round((monthly.loc[[6, 7, 8, 9]].mean() / dry - 1) * 100, 1),
            "mean_cell_temp_c": round(float(sim.loc[sim["solar_elevation"] > 10, "temp_cell"].mean()), 1),
            "default_tilt": system.resolved_tilt(r.latitude),
            "optimal_tilt": int(opt["tilt"]),
            "optimal_tilt_gain_pct": round((opt["kwh_per_kwp"] * cal.k / sy - 1) * 100, 2),
        })
        print(f"{r.city:20s} {sy:6.0f} kWh/kWp  opt tilt {int(opt['tilt'])}°", flush=True)
    df = pd.DataFrame(rows).sort_values("specific_yield_p50", ascending=False)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(ATLAS_PATH, index=False)
    print(f"\nSaved {ATLAS_PATH}")


if __name__ == "__main__":
    main()
