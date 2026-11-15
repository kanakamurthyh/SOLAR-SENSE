"""
Single source of truth for paths, physical defaults, verdict thresholds and
uncertainty constants.

Conventions used everywhere in the package:
    * Time: tz-aware UTC DatetimeIndex. Hourly values are labelled at the
      CENTRE of the averaging interval (e.g. 10:30 for the 10:00-11:00 mean).
    * Irradiance: W/m².  Temperature: °C.  Wind: m/s.  Power: kW.  Energy: kWh.
    * Capacity: plant DC rating in kWp; AC rating = kWp / dc_ac_ratio.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = ROOT_DIR / "cache"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

INDIA_LOCATIONS_PATH = DATA_DIR / "india_locations.csv"
REAL_PLANTS_PATH = DATA_DIR / "real_plants_hourly.csv"
TRAINING_SET_PATH = DATA_DIR / "training_set.parquet"
ATLAS_PATH = DATA_DIR / "india_atlas.csv"

# ─────────────────────────────────────────────────────────────
# WEATHER SOURCES
# ─────────────────────────────────────────────────────────────

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
NASA_POWER_HOURLY_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PVGIS_SERIES_URL = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"

# Feasibility uses this many complete historical years (more years -> better P90)
HISTORY_START_YEAR = 2015
HISTORY_END_YEAR = 2024
FORECAST_DAYS = 7

# ─────────────────────────────────────────────────────────────
# PHYSICAL SYSTEM DEFAULTS (utility-scale fixed-tilt in India)
# ─────────────────────────────────────────────────────────────

DEFAULT_CAPACITY_KWP = 1000.0
DEFAULT_DC_AC_RATIO = 1.25
DEFAULT_GAMMA_PDC = -0.0035        # mono-PERC power temperature coefficient, 1/°C
DEFAULT_INVERTER_EFFICIENCY = 0.975
DEFAULT_ALBEDO = 0.20

# System losses in % (pvlib.pvsystem.pvwatts_losses categories).
# Soiling is higher than the PVWatts US default because of Indian dust and
# monsoon-season cleaning gaps.
DEFAULT_LOSSES_PCT = {
    "soiling": 4.0,
    "shading": 1.0,
    "snow": 0.0,
    "mismatch": 2.0,
    "wiring": 2.0,
    "connections": 0.5,
    "lid": 1.5,
    "nameplate_rating": 1.0,
    "age": 0.0,          # first-year (new plant) energy
    "availability": 2.0,
}

# ─────────────────────────────────────────────────────────────
# VERDICT THRESHOLDS
# ─────────────────────────────────────────────────────────────

# Verdict on P50 specific yield (kWh AC per kWp DC per year).
# Indian fixed-tilt plants range roughly 1,250 (NE / coastal monsoon) to
# 1,800 (Rajasthan / Ladakh).
SPECIFIC_YIELD_VIABLE = 1500.0
SPECIFIC_YIELD_MARGINAL = 1300.0

# ─────────────────────────────────────────────────────────────
# UNCERTAINTY (1-sigma, % of annual energy)
# ─────────────────────────────────────────────────────────────

# Spread of plane-of-array irradiation between two independent ERA5
# processing chains (Open-Meteo vs PVGIS-ERA5) over 36 Indian cities
# (scripts/validate_physics_vs_pvgis.py -> reports/physics_vs_pvgis.csv).
SOURCE_UNCERTAINTY_PCT = 4.1
# Used only if models/hindcast_model_card.json is missing; otherwise the
# measured plant-to-plant spread (k_std) from real plants is used.
FALLBACK_MODEL_UNCERTAINTY_PCT = 6.0
# Forecast error growth floor per lead day for cloudier climates than the
# California plant the growth was measured on.
MIN_FORECAST_ERROR_GROWTH_PER_DAY = 0.05

# ─────────────────────────────────────────────────────────────
# APP DEFAULTS
# ─────────────────────────────────────────────────────────────

DEFAULT_LATITUDE = 26.9124         # Jaipur
DEFAULT_LONGITUDE = 75.7873
