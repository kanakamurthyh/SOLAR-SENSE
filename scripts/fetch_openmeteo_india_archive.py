"""
Fetch Open-Meteo Historical Archive (ERA5/ERA5-Land reanalysis, free, no key)
hourly weather for every site in data/india_locations.csv.

Why: the app forecasts with Open-Meteo, so training weather from the same
source family (rather than PVGIS-SARAH irradiance) removes a systematic
irradiance mismatch. Unlike PVGIS this also gives DNI and DHI, which are
needed for a proper plane-of-array transposition.

All timestamps are UTC. Irradiance is W/m² (divide by 1000 before feeding the
model). Wind is m/s.

Run from project root:
    python3 scripts/fetch_openmeteo_india_archive.py --start 2020-01-01 --end 2022-12-31
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from solarsense.config import CACHE_DIR, DATA_DIR, INDIA_LOCATIONS_PATH  # noqa: E402

URL = "https://archive-api.open-meteo.com/v1/archive"
VARS = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "cloud_cover"]


def fetch_site(row, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    dst = cache_dir / f"{row.location_id}_{start}_{end}.csv"
    if dst.exists():
        return pd.read_csv(dst)

    for _attempt in range(6):
        r = requests.get(URL, params={
            "latitude": row.latitude, "longitude": row.longitude,
            "start_date": start, "end_date": end,
            "hourly": ",".join(VARS), "timezone": "UTC", "wind_speed_unit": "ms",
        }, timeout=120)
        payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code == 200 and "hourly" in payload:
            break
        # Free tier: ~600 calls/min but multi-year hourly calls are weighted; back off.
        print(f"  {row.location_id}: HTTP {r.status_code} {str(payload)[:80]} - waiting 65 s", flush=True)
        time.sleep(65)
    else:
        raise RuntimeError(f"Open-Meteo archive failed for {row.location_id}")

    df = pd.DataFrame(payload["hourly"])
    df["LOCATION_ID"] = row.location_id
    df["LATITUDE"] = row.latitude
    df["LONGITUDE"] = row.longitude
    df.to_csv(dst, index=False)
    print(f"  {row.location_id}: {len(df):,} rows", flush=True)
    time.sleep(8)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2022-12-31")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cache_dir = CACHE_DIR / "openmeteo_archive"
    cache_dir.mkdir(parents=True, exist_ok=True)
    locations = pd.read_csv(INDIA_LOCATIONS_PATH)
    if args.limit:
        locations = locations.head(args.limit)

    print(f"Fetching Open-Meteo archive for {len(locations)} sites, {args.start}..{args.end}")
    frames = [fetch_site(row, args.start, args.end, cache_dir) for row in locations.itertuples(index=False)]
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"time": "DATE_TIME"})
    out = DATA_DIR / f"india_openmeteo_hourly_{args.start[:4]}_{args.end[:4]}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} rows, {df.LOCATION_ID.nunique()} sites -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
