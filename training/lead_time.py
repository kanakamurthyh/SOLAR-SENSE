"""
Measure how forecast error grows with lead time (day 0 .. day 6) on real plants,
using Open-Meteo's Previous Runs archive (the same target hour as forecast
N days earlier).

For each lead day: GHI forecast -> Erbs DNI/DHI -> physics -> compare with the
plant's normalised measured output. The resulting error ratios widen the
7-day forecast's uncertainty bands day by day, and are stored in the
forecast model card.

Run after training.train:
    python3 -m training.lead_time
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from solarsense.config import MODELS_DIR, REAL_PLANTS_PATH, REPORTS_DIR
from solarsense.physics import PVSystem, simulate
from solarsense.weather import WEATHER_CACHE, _finalise, _get_json, fill_dni_dhi
from training.build_training_set import ac_limit, plant_frame
from training.plants import PLANTS

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEADS = range(0, 7)
WINDOWS = {  # plant: (start, end) inside both the plant record and the archive
    "nrel_2107_arbuckle_ca": ("2022-01-01", "2024-10-31"),
    "pvdaq_1423_henderson_nv": ("2022-01-01", "2022-12-31"),
    "pvdaq_10_golden_co": ("2022-01-01", "2023-02-28"),
}


def fetch_leads(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    path = WEATHER_CACHE / f"om_prevruns_{lat:.2f}_{lon:.2f}_{start}_{end}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    names = ["shortwave_radiation"] + [f"shortwave_radiation_previous_day{d}" for d in range(1, 7)]
    payload = _get_json(URL, {
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "timezone": "UTC",
        "hourly": ",".join(names + ["temperature_2m", "wind_speed_10m"]), "wind_speed_unit": "ms",
    })
    h = payload["hourly"]
    df = pd.DataFrame({k: h[k] for k in h if k != "time"})
    df.index = pd.DatetimeIndex(pd.to_datetime(h["time"], utc=True) - pd.Timedelta(minutes=30), name="time")
    WEATHER_CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def main():
    measured_all = pd.read_csv(REAL_PLANTS_PATH)
    card_path = MODELS_DIR / "forecast_model_card.json"
    card = json.loads(card_path.read_text())
    k = card["plant_energy_factor_k"]
    rows = []
    for pid, (start, end) in WINDOWS.items():
        spec = PLANTS[pid]
        m = plant_frame(measured_all, pid)
        m = m[(m.index >= pd.Timestamp(start, tz="UTC")) & (m.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))]
        raw = fetch_leads(spec["lat"], spec["lon"], start, end)
        system = PVSystem(capacity_kwp=spec["kwp"], dc_ac_ratio=float(np.clip(spec["kwp"] / ac_limit(spec, m), 0.8, 2)),
                          tilt=spec["tilt"] if spec["tilt"] is not None else 6.0, azimuth=spec["azimuth"])
        y = m["AC_POWER"] / spec["kwp"] / k[pid]
        for lead in LEADS:
            col = "shortwave_radiation" if lead == 0 else f"shortwave_radiation_previous_day{lead}"
            w = pd.DataFrame({"ghi": raw[col], "temp_air": raw["temperature_2m"],
                              "wind_speed": raw["wind_speed_10m"]}, index=raw.index).dropna()
            if len(w) < 1000:
                continue
            w = fill_dni_dhi(w, spec["lat"], spec["lon"])
            w = _finalise(w, f"prevruns_d{lead}", spec["lat"], spec["lon"], None)
            sim = simulate(w, system, spec["lat"], spec["lon"])
            j = pd.concat([sim["ac"] / spec["kwp"], y.rename("y"), sim["solar_elevation"]], axis=1, join="inner").dropna()
            j = j[j["solar_elevation"] > 0]
            err = j.iloc[:, 0] - j["y"]
            rows.append({"plant": pid, "lead_day": lead, "n": len(j),
                         "nrmse_pct": float(np.sqrt((err ** 2).mean()) * 100),
                         "nmae_pct": float(err.abs().mean() * 100)})
        print(pid, [(r["lead_day"], round(r["nrmse_pct"], 2)) for r in rows if r["plant"] == pid], flush=True)

    df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(REPORTS_DIR / "forecast_lead_time.csv", index=False)
    rel = df.pivot(index="lead_day", columns="plant", values="nrmse_pct")
    rel = rel.div(rel.loc[0], axis=1).mean(axis=1)
    rel = rel.cummax()  # error growth should be monotone
    card["lead_day_error_ratio"] = {int(d): round(float(v), 3) for d, v in rel.items()}
    card_path.write_text(json.dumps(card, indent=2, default=str))
    print("error ratio vs day 0:", card["lead_day_error_ratio"])


if __name__ == "__main__":
    main()
