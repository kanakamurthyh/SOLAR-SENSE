"""
Build an India-wide hourly solar PV dataset from the PVGIS open API.

This script turns public PVGIS simulations into a realistic enterprise-style
solar fleet dataset: many Indian locations, multiple full years, hourly weather,
irradiance, and PV output.

Run from project root:
    python3 scripts/build_india_pvgis_dataset.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from solarsense.config import CACHE_DIR, DATA_DIR, INDIA_LOCATIONS_PATH, PVGIS_SERIES_URL  # noqa: E402

# PVGIS reference system used for the golden test (scripts/validate_physics_vs_pvgis.py).
# Note: for India PVGIS uses its ERA5 radiation database, not satellite data.
INDIA_PVGIS_PATH = DATA_DIR / "india_pvgis_hourly.csv"
PVGIS_START_YEAR = 2020
PVGIS_END_YEAR = 2022
PVGIS_SYSTEM_CAPACITY_KW = 1.0
PVGIS_SYSTEM_LOSS_PCT = 0
PVGIS_PANEL_TILT_DEG = 20
PVGIS_AZIMUTH_DEG = 0          # PVGIS convention: 0 = south
MODULE_NOCT_C = 45.0


def fetch_pvgis_location(row, start_year, end_year, timeout=(10, 90)):
    cache_dir = CACHE_DIR / "pvgis"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"{row.location_id}_{start_year}_{end_year}"
        f"_kw{PVGIS_SYSTEM_CAPACITY_KW:g}_loss{PVGIS_SYSTEM_LOSS_PCT:g}"
        f"_tilt{PVGIS_PANEL_TILT_DEG:g}_az{PVGIS_AZIMUTH_DEG:g}.json"
    )

    if cache_file.exists():
        return json.loads(cache_file.read_text())

    params = {
        "lat": float(row.latitude),
        "lon": float(row.longitude),
        "startyear": int(start_year),
        "endyear": int(end_year),
        "pvcalculation": 1,
        "peakpower": PVGIS_SYSTEM_CAPACITY_KW,
        "loss": PVGIS_SYSTEM_LOSS_PCT,
        "angle": PVGIS_PANEL_TILT_DEG,
        "aspect": PVGIS_AZIMUTH_DEG,
        "outputformat": "json",
    }
    response = requests.get(PVGIS_SERIES_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if "outputs" not in payload or "hourly" not in payload["outputs"]:
        raise ValueError(f"PVGIS response missing hourly outputs for {row.location_id}")

    temp_file = cache_file.with_suffix(".tmp")
    temp_file.write_text(json.dumps(payload))
    temp_file.replace(cache_file)
    return payload


def normalise_pvgis_payload(row, payload):
    hourly = pd.DataFrame(payload["outputs"]["hourly"])
    required = {"time", "P", "G(i)", "T2m"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"PVGIS hourly output missing columns {sorted(missing)}")

    df = pd.DataFrame()
    df["DATE_TIME"] = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", errors="coerce")
    df["AC_POWER"] = pd.to_numeric(hourly["P"], errors="coerce") / 1000.0
    df["IRRADIATION"] = pd.to_numeric(hourly["G(i)"], errors="coerce") / 1000.0
    df["AMBIENT_TEMPERATURE"] = pd.to_numeric(hourly["T2m"], errors="coerce")
    df["MODULE_TEMPERATURE"] = estimate_module_temperature(
        df["AMBIENT_TEMPERATURE"], df["IRRADIATION"]
    )
    df["LATITUDE"] = float(row.latitude)
    df["LONGITUDE"] = float(row.longitude)
    df["LOCATION_ID"] = row.location_id
    df["CITY"] = row.city
    df["STATE"] = row.state
    df["REGION"] = row.region
    df["SYSTEM_CAPACITY_KW"] = PVGIS_SYSTEM_CAPACITY_KW

    if "WS10m" in hourly.columns:
        df["WIND_SPEED_10M"] = pd.to_numeric(hourly["WS10m"], errors="coerce")
    if "H_sun" in hourly.columns:
        df["SUN_ELEVATION"] = pd.to_numeric(hourly["H_sun"], errors="coerce")

    df = df.dropna(subset=["DATE_TIME", "AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE"])
    df = df[(df["AC_POWER"] >= 0) & (df["IRRADIATION"] >= 0)]
    return df.reset_index(drop=True)


def estimate_module_temperature(ambient_c, irradiance_kw_m2, noct_c=MODULE_NOCT_C):
    irradiance_w_m2 = pd.Series(irradiance_kw_m2).clip(lower=0) * 1000.0
    return pd.Series(ambient_c) + ((noct_c - 20.0) / 800.0) * irradiance_w_m2


def build_dataset(limit=None, start_year=PVGIS_START_YEAR, end_year=PVGIS_END_YEAR, pause_s=0.2):
    locations = pd.read_csv(INDIA_LOCATIONS_PATH)
    if limit:
        locations = locations.head(int(limit))

    frames = []
    print(f"Building PVGIS India dataset: {len(locations)} locations, {start_year}-{end_year}", flush=True)
    failures = []
    for idx, row in enumerate(locations.itertuples(index=False), start=1):
        label = f"{row.city}, {row.state}"
        print(f"[{idx:02d}/{len(locations):02d}] {label}", flush=True)
        try:
            payload = fetch_pvgis_location(row, start_year, end_year)
            frames.append(normalise_pvgis_payload(row, payload))
        except Exception as exc:
            failures.append((row.location_id, str(exc)))
            print(f"  skipped: {exc}", flush=True)
        time.sleep(pause_s)

    if not frames:
        raise RuntimeError("No PVGIS locations were fetched successfully.")

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sort_values(["DATE_TIME", "LOCATION_ID"]).reset_index(drop=True)
    INDIA_PVGIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(INDIA_PVGIS_PATH, index=False)
    print(f"\nSaved {len(dataset):,} rows -> {INDIA_PVGIS_PATH}")
    print(
        f"Coverage: {dataset['LOCATION_ID'].nunique()} locations, "
        f"{dataset['DATE_TIME'].min()} to {dataset['DATE_TIME'].max()}"
    )
    if failures:
        print("\nSkipped locations:")
        for location_id, reason in failures:
            print(f"  - {location_id}: {reason}")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N locations")
    parser.add_argument("--start-year", type=int, default=PVGIS_START_YEAR)
    parser.add_argument("--end-year", type=int, default=PVGIS_END_YEAR)
    args = parser.parse_args()
    build_dataset(args.limit, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
