"""
Add multi-year, hourly, REAL PV generation from NREL PVDAQ public systems
(OEDI bucket, no API key) to data/real_plants_hourly.csv.

Default systems (all with metered AC power + on-site POA irradiance + ambient temp):
    34    Las Vegas NV  146.6 kW  fixed 11°/180°  Köppen BWh (hot desert, Rajasthan-like)  2015-2020
    35    Las Vegas NV  gym roof  fixed, az 270°  BWh                                     2015-2020
    1423  Henderson NV    6.0 kW  fixed 35°/180°  BWh, 2 inverters, module temps          2019-2022
    10    Golden CO      1.12 kW  fixed 40°/180°  BSk (cold semi-arid, Leh-like winters)  2020-2023

Schema is identical to scripts/build_real_plant_dataset.py (UTC, kW, kW/m², °C, m/s).
Rows for a LOCATION_ID are replaced on re-run, other plants are kept.

Run from project root:
    python3 scripts/build_pvdaq_dataset.py [--systems 34 35] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from scripts.build_real_plant_dataset import OUT_PATH, shift_to_utc, to_celsius, to_ms  # noqa: E402
from solarsense.config import CACHE_DIR  # noqa: E402

BASE = "https://oedi-data-lake.s3.amazonaws.com/"
RAW_DIR = CACHE_DIR / "pvdaq"

SYSTEMS = {
    34:   dict(location_id="pvdaq_34_las_vegas_nv", years=range(2015, 2021), mount="fixed_11deg"),
    35:   dict(location_id="pvdaq_35_las_vegas_nv", years=range(2015, 2021), mount="fixed_az270"),
    1423: dict(location_id="pvdaq_1423_henderson_nv", years=range(2019, 2023), mount="fixed_35deg"),
    10:   dict(location_id="pvdaq_10_golden_co", years=range(2020, 2024), mount="fixed_40deg"),
}


# ─────────────────────────────────────────────────────────────
# S3 LISTING + DOWNLOAD
# ─────────────────────────────────────────────────────────────

def list_keys(prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        xml = requests.get(BASE, params=params, timeout=60).text
        keys += re.findall(r"<Key>([^<]+\.csv)</Key>", xml)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def download(key: str) -> Path:
    dst = RAW_DIR / key.split("pvdata/")[-1]
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(BASE + key, timeout=120)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return dst
        except requests.RequestException as e:
            if attempt == 2:
                raise
            print(f"    retry {key.split('/')[-1]}: {e}", flush=True)
    return dst


def system_metadata(sid: int) -> dict:
    dst = RAW_DIR / f"{sid}_system_metadata.json"
    if not dst.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(requests.get(f"{BASE}pvdaq/csv/system_metadata/{sid}_system_metadata.json", timeout=60).content)
    meta = json.loads(dst.read_text())
    site = meta.get("Site", {})
    if isinstance(site, list):
        site = site[0] if site else {}
    mount = meta.get("Mount", {})
    if isinstance(mount, list):
        mount = mount[0] if mount else {}
    return {
        "name": meta["System"].get("public_name", str(sid)),
        "kwp": float(meta["System"].get("power") or "nan"),
        "lat": float(site.get("latitude") or "nan"),
        "lon": float(site.get("longitude") or "nan"),
        "tilt": mount.get("tilt"), "azimuth": mount.get("azimuth"),
    }


# ─────────────────────────────────────────────────────────────
# NORMALISE ONE SYSTEM
# ─────────────────────────────────────────────────────────────

def cols_like(df: pd.DataFrame, pattern: str, exclude: str | None = None) -> list[str]:
    out = [c for c in df.columns if re.search(pattern, c, re.I)]
    if exclude:
        out = [c for c in out if not re.search(exclude, c, re.I)]
    return out


def normalise_power_units(ac: pd.Series, kwp: float) -> pd.Series:
    """
    PVDAQ loggers report power in W, hW ("_hw" columns) or kW, and some sites
    switch units between years (system 34 went hW -> W in 2018). Per calendar
    year, pick the power-of-ten scale that puts the sustained peak (P99.5) at
    ~0.8 x nameplate, i.e. a normal AC/DC ratio.
    """
    if not np.isfinite(kwp) or kwp <= 0:
        return ac
    out = ac.copy()
    for yr, s in ac.groupby(ac.index.year):
        p = s.quantile(0.995)
        if not np.isfinite(p) or p <= 0:
            continue
        scale = min((1.0, 1e-1, 1e-2, 1e-3), key=lambda f: abs(np.log10(p * f / (0.8 * kwp))))
        if scale != 1.0:
            print(f"    {yr}: AC peak {p:,.0f} vs {kwp:.1f} kWdc -> scaling x{scale:g}")
        out.loc[s.index] = s * scale
    return out


def build_system(sid: int, cfg: dict, workers: int) -> pd.DataFrame:
    meta = system_metadata(sid)
    print(f"[{sid}] {meta['name']}  {meta['kwp']:.2f} kWdc  ({meta['lat']:.4f}, {meta['lon']:.4f})")

    keys = []
    for yr in cfg["years"]:
        keys += list_keys(f"pvdaq/csv/pvdata/system_id={sid}/year={yr}/")
    print(f"  {len(keys)} day-files, downloading with {workers} workers...", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        paths = list(ex.map(download, keys))

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
        except Exception as e:  # a few day files are empty/corrupt
            print(f"    skip {p.name}: {e}")
    raw = pd.concat(frames, ignore_index=True)
    raw["measured_on"] = pd.to_datetime(raw["measured_on"], errors="coerce")
    raw = raw.dropna(subset=["measured_on"]).drop_duplicates("measured_on").set_index("measured_on").sort_index()

    # AC power: sum inverter/meter AC channels (some systems report per inverter)
    ac_cols = cols_like(raw, r"(^|_)ac_power", exclude=r"dc_|factor")
    if not ac_cols:
        raise ValueError(f"{sid}: no ac_power column in {list(raw.columns)}")
    ac = raw[ac_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
    ac = normalise_power_units(ac, meta["kwp"])
    p995 = float(ac.quantile(0.995))
    ac = ac.where((ac >= -0.02 * p995) & (ac <= 1.25 * p995)).clip(lower=0)

    poa_cols = cols_like(raw, r"poa_irradiance", exclude=r"refcell")
    if not poa_cols:
        raise ValueError(f"{sid}: no poa_irradiance column")
    poa = pd.to_numeric(raw[poa_cols[0]], errors="coerce")
    poa = poa.where((poa > -10) & (poa < 1500)).clip(lower=0)

    hourly = pd.DataFrame({
        "AC_POWER": ac.resample("1h").mean(),
        "IRRADIATION": poa.resample("1h").mean() / 1000.0,
    })
    t_cols = cols_like(raw, r"ambient_temp")
    if t_cols:
        t = pd.to_numeric(raw[t_cols[0]], errors="coerce")
        t = t.where((t > -60) & (t < 130))
        hourly["AMBIENT_TEMPERATURE"] = to_celsius(t).resample("1h").mean()
    m_cols = cols_like(raw, r"module_temp")
    if m_cols:
        mt = raw[m_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        hourly["MODULE_TEMPERATURE"] = to_celsius(mt.where((mt > -60) & (mt < 200))).resample("1h").mean()
    w_cols = cols_like(raw, r"wind_speed")
    if w_cols:
        w = pd.to_numeric(raw[w_cols[0]], errors="coerce")
        hourly["WIND_SPEED_10M"] = to_ms(w.where((w >= 0) & (w < 80)), w_cols[0]).resample("1h").mean()

    hourly = shift_to_utc(hourly, meta["lon"])

    hourly = hourly.dropna(subset=["AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE"])

    # QC: drop months whose daytime output collapses (outages / logger faults)
    day = hourly[hourly["IRRADIATION"] > 0.05]
    ratio = (day.groupby(day.index.to_period("M"))["AC_POWER"].mean()
             / day.groupby(day.index.to_period("M"))["IRRADIATION"].mean())
    good = ratio[ratio > 0.25 * ratio.median()].index
    dropped = sorted(set(ratio.index) - set(good))
    if dropped:
        print(f"  QC dropped {len(dropped)} low-output months: {[str(m) for m in dropped]}")
    hourly = hourly[hourly.index.to_period("M").isin(good)]

    kwp, cap_source = meta["kwp"], "metadata_nameplate"
    if not np.isfinite(kwp) or p995 < 0.5 * kwp:
        kwp = float(hourly["AC_POWER"].sum() / (0.78 * hourly["IRRADIATION"].sum()))
        cap_source = "estimated_from_PR0.78"
        print(f"  metered peak {p995:.1f} kW vs nameplate {meta['kwp']} -> estimated {kwp:.1f} kWp")

    hourly["AC_POWER_PER_KWP"] = hourly["AC_POWER"] / kwp
    hourly["CAPACITY_SOURCE"] = cap_source
    hourly["IRRADIANCE_TYPE"] = "POA"
    hourly["LATITUDE"] = meta["lat"]
    hourly["LONGITUDE"] = meta["lon"]
    hourly["LOCATION_ID"] = cfg["location_id"]
    hourly["SYSTEM_CAPACITY_KW"] = round(kwp, 2)
    hourly["MOUNT"] = cfg["mount"]
    hourly = hourly.rename_axis("DATE_TIME").reset_index()

    yrs = hourly.groupby(hourly["DATE_TIME"].dt.year)["AC_POWER"].sum() / kwp
    print(f"  {len(hourly):,} hourly rows {hourly.DATE_TIME.min().date()} .. {hourly.DATE_TIME.max().date()}")
    print(f"  specific yield kWh/kWp by year: {yrs.round(0).astype(int).to_dict()}")
    return hourly


def merge_into_dataset(new: pd.DataFrame) -> pd.DataFrame:
    if OUT_PATH.exists():
        old = pd.read_csv(OUT_PATH, parse_dates=["DATE_TIME"])
        old = old[~old["LOCATION_ID"].isin(new["LOCATION_ID"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["LOCATION_ID", "DATE_TIME"])
    new.to_csv(OUT_PATH, index=False)
    return new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="*", type=int, default=list(SYSTEMS))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    frames = [build_system(s, SYSTEMS[s], args.workers) for s in args.systems]
    df = merge_into_dataset(pd.concat(frames, ignore_index=True))
    print(f"\nDataset now {len(df):,} rows, {df.LOCATION_ID.nunique()} plants -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")
    print(df.groupby("LOCATION_ID")["DATE_TIME"].agg(["min", "max", "count"]).to_string())


if __name__ == "__main__":
    main()
