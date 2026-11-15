"""
Build a multi-year, hourly, REAL-WORLD PV generation dataset from the NREL
2023 Solar Data Prize sites (public OEDI bucket, no API key).

Sites (all with metered AC output + on-site irradiance):
    2105  Maui, HI        110 kWdc   fixed tilt      GHI + DNI + T + wind   2018-12 .. 2023-11  (opt-in: meter channel is not PV)
    2107  Arbuckle, CA    893 kWdc   fixed 25°/180°  POA + T + wind         2017-01 .. 2024-11
    9068  Kersey, CO     4738 kWdc   1-axis tracker  POA (+ Open-Meteo T)   2024-01 .. 2025-04  (opt-in, noisy meter)

Output schema (matches src/utils/preprocessing.py expectations, UTC timestamps):
    DATE_TIME, AC_POWER (kW), AC_POWER_PER_KWP, IRRADIATION (kW/m²), IRRADIANCE_TYPE (GHI|POA),
    DNI (kW/m², where measured), AMBIENT_TEMPERATURE (°C), WIND_SPEED_10M (m/s),
    LATITUDE, LONGITUDE, LOCATION_ID, SYSTEM_CAPACITY_KW (kWp behind the meter),
    CAPACITY_SOURCE (metadata_nameplate | estimated_from_PR0.78), MOUNT

Caveats:
    * Timestamps in the source files are in a fixed local offset; the offset is
      inferred from the daylight window and removed, accurate to about +-1 h.
    * 2105's meter covers only a sub-array of the 110 kW site; its kWp is estimated.
    * IRRADIATION is GHI for 2105 and plane-of-array for 2107 (see IRRADIANCE_TYPE).

Run from project root:
    python3 scripts/build_real_plant_dataset.py [--raw-dir cache/nrel_prize]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from solarsense.config import CACHE_DIR, DATA_DIR  # noqa: E402

OEDI = "https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/"
OUT_PATH = DATA_DIR / "real_plants_hourly.csv"

SITES = {
    "2105": dict(
        location_id="nrel_2105_maui_hi", lat=20.884134, lon=-156.340543, kwp=110.0, mount="fixed",
        files={
            "meter": ["2105_meter_data.csv"],
            "irr":   ["2105_irradiance_data.csv"],
            "env":   ["2105_environment_1_data.csv", "2105_environment_2_data.csv"],
        },
    ),
    "2107": dict(
        location_id="nrel_2107_arbuckle_ca", lat=38.996306, lon=-122.134111, kwp=893.0, mount="fixed_25deg",
        files={
            "meter": ["2107_meter_15m_data.csv", "2107_meter_15m_data_2024.csv"],
            "irr":   ["2107_irradiance_data.csv", "2107_irradiance_data_2024.csv"],
            "env":   ["2107_environment_data.csv", "2107_environment_data_2024.csv"],
        },
    ),
    "9068": dict(
        location_id="nrel_9068_kersey_co", lat=40.3864, lon=-104.5512, kwp=4738.0, mount="tracker_1axis",
        files={
            "meter": ["9068_meter_data_20240101_20250430.csv"],
            "irr":   ["9068_irradiance_data_20240101_20250430.csv"],
            "env":   [],  # no compact env file -> filled from Open-Meteo archive
        },
    ),
}


# ─────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────

def fetch(site: str, name: str, raw_dir: Path) -> Path:
    dst = raw_dir / name
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{OEDI}{site}_OEDI/data/{name}"
    print(f"  fetching {name}", flush=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    return dst


def load_concat(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["measured_on"]) for p in paths]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("measured_on").sort_values("measured_on")
    return df.set_index("measured_on")


# ─────────────────────────────────────────────────────────────
# COLUMN PICKING (files use long sensor-tagged names)
# ─────────────────────────────────────────────────────────────

def pick(df: pd.DataFrame, pattern: str) -> pd.Series | None:
    cols = [c for c in df.columns if re.search(pattern, c, re.I)]
    if not cols:
        return None
    s = pd.to_numeric(df[cols[0]], errors="coerce")
    return s


def to_celsius(s: pd.Series) -> pd.Series:
    # Files do not always say; a max above 60 can only be Fahrenheit.
    return (s - 32.0) * 5.0 / 9.0 if s.max() > 60 else s


def to_ms(s: pd.Series, colname: str) -> pd.Series:
    return s * 0.44704 if "mph" in colname.lower() else s


# ─────────────────────────────────────────────────────────────
# TIMEZONE: files are in a fixed local offset, not UTC. Infer it
# from the daylight window centre vs solar noon at that longitude.
# ─────────────────────────────────────────────────────────────

def infer_utc_offset_hours(irr_hourly: pd.Series, lon: float) -> int:
    prof = irr_hourly.groupby(irr_hourly.index.hour).mean()
    lit = prof[prof > 0.05 * prof.max()].index.to_numpy()
    # handle a window that wraps midnight (e.g. UTC-labelled Pacific data)
    if lit.min() == 0 and lit.max() == 23:
        lit = np.where(lit < 12, lit + 24, lit)
    centre_local = (lit.min() + lit.max()) / 2.0 + 0.5
    solar_noon_utc = 12.0 - lon / 15.0
    offset = round(centre_local - solar_noon_utc)
    return int(((offset + 12) % 24) - 12)


def shift_to_utc(hourly: pd.DataFrame, lon: float, min_hours: int = 24 * 60) -> pd.DataFrame:
    """
    Loggers change their timestamp base between years (PVDAQ system 10 does
    in 2022), so infer and remove the offset per calendar year. Years with too
    little daylight data fall back to the offset inferred from all years.
    """
    irr = hourly["IRRADIATION"].dropna()
    global_off = infer_utc_offset_hours(irr, lon)
    parts = []
    for yr, part in hourly.groupby(hourly.index.year):
        irr_y = part["IRRADIATION"].dropna()
        off = infer_utc_offset_hours(irr_y, lon) if len(irr_y) >= min_hours else global_off
        print(f"    {yr}: timestamp offset UTC{off:+d}h -> shifted to UTC")
        part = part.copy()
        part.index = part.index - pd.Timedelta(hours=off)
        parts.append(part)
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="first")]

    # Clock sanity: after the shift every month's irradiance peak must sit within
    # 3 h of solar noon (fixed tilt/azimuth and low winter sun can move the peak
    # by ~2 h). Months that fail (logger clock faults, e.g. PVDAQ system 10 in
    # May-Jul 2022) or have <200 daylight hours are dropped rather than
    # mis-taught to the model.
    solar_noon_utc = (12.0 - lon / 15.0) % 24
    day = out[out["IRRADIATION"] > 0.05]
    peak = day.groupby([day.index.year, day.index.month, day.index.hour])["IRRADIATION"].mean()
    peak_hour = peak.groupby(level=[0, 1]).idxmax().map(lambda t: t[2])
    n_day = day.groupby([day.index.year, day.index.month]).size()
    diff = (peak_hour - solar_noon_utc + 12) % 24 - 12
    bad = sorted(set(diff[diff.abs() > 3].index) | set(n_day[n_day < 200].index))
    if bad:
        print(f"    clock/coverage QC dropped months: {[f'{y}-{m:02d}' for y, m in bad]}")
        mask = pd.Series(list(zip(out.index.year, out.index.month, strict=True)), index=out.index).isin(bad)
        out = out[~mask.values]
    return out


# ─────────────────────────────────────────────────────────────
# OPEN-METEO ARCHIVE FALLBACK (ambient temp / wind when site lacks them)
# ─────────────────────────────────────────────────────────────

def openmeteo_env(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "temperature_2m,wind_speed_10m", "timezone": "UTC", "wind_speed_unit": "ms",
    }, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"AMBIENT_TEMPERATURE": h["temperature_2m"], "WIND_SPEED_10M": h["wind_speed_10m"]},
                      index=pd.to_datetime(h["time"]))
    return df


# ─────────────────────────────────────────────────────────────
# BUILD ONE SITE
# ─────────────────────────────────────────────────────────────

def build_site(site: str, cfg: dict, raw_dir: Path) -> pd.DataFrame:
    print(f"[{site}] {cfg['location_id']}")
    meter = load_concat([fetch(site, f, raw_dir) for f in cfg["files"]["meter"]])
    irr   = load_concat([fetch(site, f, raw_dir) for f in cfg["files"]["irr"]])
    env   = load_concat([fetch(site, f, raw_dir) for f in cfg["files"]["env"]]) if cfg["files"]["env"] else None

    ac = pick(meter, r"ac_output|ac_power_\(kw\)|ac_power")
    if ac is None:
        raise ValueError(f"{site}: no AC power column in {list(meter.columns)}")
    # Meter spikes far above the sustained peak are telemetry glitches. Use the
    # metered P99.5 (not nameplate) as the reference, because some meters only
    # cover a sub-array of the plant listed in the metadata.
    p995 = float(ac.quantile(0.995))
    ac = ac.where((ac >= -0.02 * p995) & (ac <= 1.25 * p995)).clip(lower=0)

    ghi = pick(irr, r"ghi")
    poa = pick(irr, r"poa")
    dni = pick(irr, r"dni")
    if ghi is not None:
        irr_main, irr_type = ghi, "GHI"
    elif poa is not None:
        irr_main, irr_type = poa, "POA"
    else:
        raise ValueError(f"{site}: no irradiance column in {list(irr.columns)}")
    irr_main = irr_main.where((irr_main > -10) & (irr_main < 1500)).clip(lower=0)

    hourly = pd.DataFrame({
        "AC_POWER": ac.resample("1h").mean(),
        "IRRADIATION": irr_main.resample("1h").mean() / 1000.0,
    })
    if dni is not None:
        hourly["DNI"] = dni.where((dni > -10) & (dni < 1500)).clip(lower=0).resample("1h").mean() / 1000.0

    if env is not None:
        t = pick(env, r"temp")
        w_col = next((c for c in env.columns if re.search(r"wind_speed", c, re.I)), None)
        if t is not None:
            hourly["AMBIENT_TEMPERATURE"] = to_celsius(t.where((t > -60) & (t < 130))).resample("1h").mean()
        if w_col is not None:
            w = pd.to_numeric(env[w_col], errors="coerce")
            hourly["WIND_SPEED_10M"] = to_ms(w.where((w >= 0) & (w < 80)), w_col).resample("1h").mean()

    # local fixed offset -> UTC (per year)
    hourly = shift_to_utc(hourly, cfg["lon"])

    if "AMBIENT_TEMPERATURE" not in hourly.columns:
        start, end = hourly.index.min().strftime("%Y-%m-%d"), hourly.index.max().strftime("%Y-%m-%d")
        print(f"  no on-site temperature; filling from Open-Meteo archive {start}..{end}")
        hourly = hourly.join(openmeteo_env(cfg["lat"], cfg["lon"], start, end), how="left")

    hourly = hourly.dropna(subset=["AC_POWER", "IRRADIATION", "AMBIENT_TEMPERATURE"])

    # QC: drop calendar months whose daytime output collapses (outages, meter
    # direction faults) so they do not teach the model that sun -> no power.
    day = hourly[hourly["IRRADIATION"] > 0.05]
    month_ratio = (day.groupby(day.index.to_period("M"))["AC_POWER"].mean()
                   / day.groupby(day.index.to_period("M"))["IRRADIATION"].mean())
    good_months = month_ratio[month_ratio > 0.25 * month_ratio.median()].index
    dropped = sorted(set(month_ratio.index) - set(good_months))
    if dropped:
        print(f"  QC dropped {len(dropped)} low-output months: {[str(m) for m in dropped]}")
    hourly = hourly[hourly.index.to_period("M").isin(good_months)]

    # Capacity actually behind the meter. If the metered peak is far below the
    # nameplate the meter covers a sub-array; estimate kWp from energy / (PR x
    # irradiance) with a typical PR of 0.78 and flag it.
    kwp, cap_source = cfg["kwp"], "metadata_nameplate"
    if p995 < 0.5 * cfg["kwp"]:
        kwp = float(hourly["AC_POWER"].sum() / (0.78 * hourly["IRRADIATION"].sum()))
        cap_source = "estimated_from_PR0.78"
        print(f"  metered peak {p995:.0f} kW << nameplate {cfg['kwp']:.0f} kWdc -> "
              f"meter covers a sub-array; estimated {kwp:.1f} kWp ({cap_source})")
    hourly["AC_POWER_PER_KWP"] = hourly["AC_POWER"] / kwp
    hourly["CAPACITY_SOURCE"] = cap_source
    hourly["IRRADIANCE_TYPE"] = irr_type
    hourly["LATITUDE"] = cfg["lat"]
    hourly["LONGITUDE"] = cfg["lon"]
    hourly["LOCATION_ID"] = cfg["location_id"]
    hourly["SYSTEM_CAPACITY_KW"] = round(kwp, 1)
    hourly["MOUNT"] = cfg["mount"]
    hourly = hourly.rename_axis("DATE_TIME").reset_index()

    yrs = hourly.groupby(hourly["DATE_TIME"].dt.year)["AC_POWER"].sum() / kwp
    print(f"  {len(hourly):,} hourly rows {hourly.DATE_TIME.min().date()} .. {hourly.DATE_TIME.max().date()}")
    print(f"  specific yield kWh/kWp by year: {yrs.round(0).astype(int).to_dict()}")
    return hourly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(CACHE_DIR / "nrel_prize"))
    # Opt-in sites: 2105's public "meter_ac_output" channel is flat ~5.6 kW day
    # and night (not PV output); 9068's meter has ~35% negative readings and
    # whole months of reversed/zero output that QC cannot fully repair.
    parser.add_argument("--sites", nargs="*", default=["2107"],
                        help=f"subset of {list(SITES)}; 2105 and 9068 are low quality")
    args = parser.parse_args()
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames = [build_site(s, SITES[s], raw_dir) for s in args.sites]
    df = pd.concat(frames, ignore_index=True).sort_values(["LOCATION_ID", "DATE_TIME"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df):,} rows, {df.LOCATION_ID.nunique()} plants -> {OUT_PATH} "
          f"({OUT_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
