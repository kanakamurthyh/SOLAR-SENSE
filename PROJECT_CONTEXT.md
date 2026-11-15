# SolarSense: Project Context (v2)

> Snapshot as of 2 October 2026, after the v2 rebuild. It covers how the project works, the pipeline, the data, the models, an honest quality assessment, what is still open and what to do next.
> Every number here was measured by running the code and data in this repository.

---

## Contents

1. [What the project is](#1-what-the-project-is)
2. [What changed from v1](#2-what-changed-from-v1)
3. [Architecture and pipelines](#3-architecture-and-pipelines)
4. [File-by-file reference](#4-file-by-file-reference)
5. [Data](#5-data)
6. [Models: physics, calibration, ML](#6-models)
7. [Validation results](#7-validation-results)
8. [Current state](#8-current-state)
9. [How good is it now?](#9-how-good-is-it-now)
10. [Remaining problems and how to tackle them](#10-remaining-problems-and-how-to-tackle-them)
11. [Roadmap](#11-roadmap)
12. [How to run and rebuild](#12-how-to-run-and-rebuild)
13. [Glossary](#13-glossary)

---

## 1. What the project is

SolarSense is a Streamlit app plus a Python package (`solarsense/`) for solar decisions in India:

| Tab | Question | Output |
|---|---|---|
| ☀ Feasibility | How much energy will this site produce? | P50/P90 annual energy, specific yield, CUF, performance ratio, loss waterfall, year-to-year variability, uncertainty budget, energy verdict with reasons, tilt optimiser, JSON report |
| 🌤 7-Day Forecast | What will my plant produce? | Hourly P10/P50/P90 fan chart (local time), daily bars with calibrated bands, cloud cover |
| 🗺 India Atlas | Where is best? | 36-city map and ranking, regional averages, monthly-profile comparison; click a city to load it |
| 🔬 Model & Validation | Can I trust it? | Real-plant calibration, leave-one-plant-out metrics, lead-time curve, golden test vs PVGIS, measured-vs-physics time series |
| 🤖 AI Analyst | Natural-language analysis | Claude (tool use) calls the engines above and only quotes numbers they return (optional; needs an API key) |

---

## 2. What changed from v1

### 2.1 Every v1 defect and its resolution

| v1 problem | Resolution in v2 |
|---|---|
| 🔴 Forecast fed W/m² to a kW/m² model (2.4× overestimate, flat curve) | All weather enters through `solarsense/weather.py` as **W/m²**, and `validate_weather()` rejects kW/m² frames. Test: `test_rejects_kw_per_m2`, `test_forecast_units_bands_and_local_time` |
| 🔴 Forecast in local time vs a UTC-trained model | Every source is requested in **UTC** and labelled at the interval centre. Display converts to IST. Test: `test_peak_power_near_solar_noon_utc` |
| 🔴 *(found during v2)* **NASA POWER cache was in local solar time**, not UTC (the API default; v1 never set `time-standard`). So v1's feasibility path was also shifted by about 5 h. The original review wrongly said this path was consistent. | NASA requests now send `time-standard=UTC` and use a new cache key |
| 🔴 "Skill score 96.6 %" compared different cities at the same hour | Removed. The forecast is now scored against **24 h persistence per plant** on held-out plants: 36 % skill |
| 🔴 Feasibility optimistic: 0 % losses, 7-knob fudge derate, DC/AC confusion | Fudge derate deleted. pvlib physics with an explicit Indian loss chain (≈13.2 %), DC/AC ratio input, inverter clipping, and a **measured real-plant calibration** (k = 0.964) |
| 🟠 Model only imitated a simulator (R² 0.9994 meaningless) | ML now learns from **real metered plants**. When it couldn't beat physics on point accuracy, it was given the job it *can* do: calibrated uncertainty |
| 🟠 Single split, test set covered Jul–Dec only, no spatial CV | **Leave-one-plant-out** CV across 5 plants and 3 climates, multi-year |
| 🟠 Redundant / location-memorising features | Features are physically transferable only: **no lat/lon/month/plant id** (`test_feature_frame_contract`) |
| 🟠 Results vanished on any widget change | Results live in `st.session_state` with a "stale inputs" banner |
| 🟠 Naive finance (cost ÷ savings) | **Removed.** SolarSense is energy-only by design (decision of 2 Oct 2026): no costs, tariffs or returns anywhere in the code or UI |
| 🟡 No tests / CI / pins, pickle model, `sys.path` hacks, binaries in git, dead code | 29 offline tests, GitHub Actions (ruff + pytest), pinned `requirements.txt`, `pyproject.toml`, XGBoost `.ubj` (no pickle), package imports, `.gitignore`, Makefile, Dockerfile. Old `src/` removed |

### 2.2 Second correction to the original review
The review said PVGIS uses satellite irradiance. **For India, PVGIS uses its ERA5 database** (`radiation_db: PVGIS-ERA5`). The v1 "simulated training data" and the Open-Meteo archive therefore share the same underlying reanalysis, processed differently. Their disagreement (4.1 % σ) is processing-chain uncertainty, not error against ground truth.

### 2.3 Where v1 code went
The full pre-upgrade working tree (including the uncommitted v1 edits) is saved in `.backup/pre-upgrade-2026-10-02.tar.gz` (gitignored). The old Kaggle EDA notebook moved to `notebooks/eda_kaggle_plants.ipynb`.

---

## 3. Architecture and pipelines

### 3.1 Design principles
1. **Physics is the backbone.** pvlib gives the central (P50) estimate and respects the user's plant and loss settings.
2. **Real data calibrates and bounds it.** The measured ÷ physics ratio across real plants corrects optimism, and its spread sets the P90.
3. **ML quantifies uncertainty.** Quantile XGBoost half-widths, scaled by conformal calibration on held-out plants, give bands that cover 80 % of real outcomes.
4. **One unit and time contract.** UTC, interval-centre labels, W/m², °C, m/s, kW, kWh. It is enforced at a single boundary (`weather.py`).
5. **The LLM orchestrates and explains. It never computes numbers.**

### 3.2 Feasibility pipeline (`solarsense/feasibility.py`)
```
get_history(lat, lon, 2015, 2024)          Open-Meteo ERA5 hourly (GHI, DNI, DHI, T, wind, RH, cloud) → NASA POWER fallback
  → complete_years()                       keep full calendar years only
  → simulate(weather, PVSystem)            pvlib chain, user losses, DC/AC clipping
  → × k (hindcast model card)              real-plant calibration, 0.964
  → annual sum per year                    P50 = mean of 10 years; σ_iav = year-to-year spread
  → σ_LT = √(σ_model² + σ_source² + σ_iav²/10)       σ_model = k spread 5.2 %, σ_source 4.1 %
    P90 = P50 · (1 − 1.2816 σ_LT)          also single-year P90
  → monthly P50 + 10th/90th-percentile year, PR, climate summary, loss waterfall
  → make_verdict()
```
**Verdict rules** (`solarsense/config.py`), on P50 specific yield:

| VIABLE | MARGINAL | NOT VIABLE |
|---|---|---|
| ≥ 1,500 kWh/kWp/yr | ≥ 1,300 kWh/kWp/yr | below 1,300 |

A caution is added when the verdict is VIABLE or MARGINAL but P90 falls below 1,300.

### 3.3 Forecast pipeline (`solarsense/forecast.py`)
```
fetch_openmeteo_forecast(lat, lon)         7 days hourly NWP, UTC
  → simulate(...) with user losses → × k_forecast (0.940)              = P50
  → ML quantile half-widths (ideal-physics features)
      × Mondrian conformal scale for the hour's clear-sky-index bin
      × lead-day factor max(measured, 1 + 0.05·day)                   = P10 / P90 (clipped to 0 … AC rating)
  → daily sums; daily bands use separately calibrated daily scales
  → local time (Asia/Kolkata inside India)
```

### 3.4 Training pipeline (`training/`)
```
build_training_set.py   for each real plant (training/plants.py):
                          ERA5 archive (hindcast) + archived NWP forecasts 2021+ (forecast)
                          → fit missing tilt from the plant's POA sensor (gym roof → 6°)
                          → per-year clock alignment (±2 h search vs physics POA)
                          → physics with the plant's own geometry → ML features + measured target
                        → data/training_set.parquet (99,909 daylight hours)
train.py                y = k_plant × aging(t) × z(t)
                          k_plant: measured ÷ physics (age- and nameplate-corrected)
                          z: normalised hourly shape → quantile XGBoost
                        leave-one-plant-out CV vs physics and 24 h persistence
                        Mondrian conformal calibration (clear-sky-index bins), hourly and daily
                        → models/{forecast,hindcast}_model.ubj + model cards, reports/
lead_time.py            Open-Meteo Previous Runs (day 0…6) → error growth ratio → forecast model card
```

### 3.5 AI Analyst (`solarsense/agent.py`)
A manual tool-use loop on `claude-opus-5-5` (effort `medium`, server-side refusal fallback `fallbacks="default"`), at most 8 tool rounds. It has five tools: `find_city`, `analyse_site`, `compare_sites`, `atlas_lookup`, `forecast_production`. Tool errors come back as `is_error` results. The system prompt requires every number to come from a tool result. The tab shows setup instructions when no credentials are present.

---

## 4. File-by-file reference

| Path | Purpose |
|---|---|
| `solarsense/config.py` | Paths, API URLs, history window, physical defaults (DC/AC 1.25, γ −0.35 %/°C, Indian loss chain), verdict thresholds, uncertainty constants |
| `solarsense/weather.py` | WeatherFrame contract + `validate_weather`; Open-Meteo archive / forecast / historical forecast; NASA POWER (UTC) with Erbs decomposition; parquet cache; retries/backoff; `get_history` with fallback |
| `solarsense/physics.py` | `PVSystem` dataclass; `simulate()` (Perez, IAM, Faiman, PVWatts, losses, clipping, clear-sky index); `loss_waterfall()`; `optimise_orientation()` |
| `solarsense/ml.py` | 15 transferable features; `QuantileModel` (predict / save / load, monotone and capped); `fit()`; `score()` incl. pinball and coverage |
| `solarsense/calibration.py` | Reads model cards: k, k_std, conformal scales per clear-sky bin, lead-day factors |
| `solarsense/feasibility.py` | `run_feasibility()`, `make_verdict()`, `best_orientation()` |
| `solarsense/forecast.py` | `run_forecast()`, `local_timezone()` |
| `solarsense/agent.py` | Claude tool definitions, tool handlers, agent loop, availability check |
| `training/plants.py` | Real-plant registry: coordinates, kWp, tilt/azimuth, AC limit, technology, commissioning date |
| `training/build_training_set.py`, `train.py`, `lead_time.py` | See §3.4 |
| `scripts/build_real_plant_dataset.py` | NREL Solar Data Prize site 2107 → `data/real_plants_hourly.csv` (writes fresh) |
| `scripts/build_pvdaq_dataset.py` | NREL PVDAQ systems 34, 35, 1423, 10 → merged into the same CSV (run after the above) |
| `scripts/fetch_openmeteo_india_archive.py` | ERA5 hourly for the 36 cities |
| `scripts/build_india_pvgis_dataset.py` | PVGIS reference output for the 36 cities (golden test) |
| `scripts/validate_physics_vs_pvgis.py` | Golden test → `reports/physics_vs_pvgis.csv` |
| `scripts/build_india_atlas.py` | → `data/india_atlas.csv` |
| `app/app.py`, `app/style.css` | Streamlit UI (Plotly charts, session state, cached weather) |
| `tests/` | 29 offline tests + `-m network` smoke test |
| `Makefile`, `pyproject.toml`, `requirements.txt`, `Dockerfile`, `.github/workflows/ci.yml` | Tooling |

---

## 5. Data

| File | Real / simulated | Rows | Used for |
|---|---|---|---|
| `data/real_plants_hourly.csv` | **Real, metered** (NREL) | 141,891 h, 5 plants, 2015–2024 | Training, calibration, validation |
| `data/training_set.parquet` | Real targets + reanalysis/NWP features | 99,909 daylight h | ML training (gitignored, `make train`) |
| `data/india_openmeteo_hourly_2020_2022.csv` | Reanalysis (ERA5) | 946,944 | Atlas, golden test (gitignored) |
| `data/india_pvgis_hourly.csv` | Simulated (PVGIS on ERA5) | 946,944 | Golden test reference (gitignored) |
| `data/india_atlas.csv` | Derived | 36 | Atlas tab, agent |
| `data/india_locations.csv` | Curated | 36 | City presets |
| Kaggle `Plant_*` CSVs | Real, 34 days, 2 Indian plants | ~136 k raw | Not used (location unknown → no weather match); kept for EDA |
| Runtime: ERA5 2015–2024 per site | Reanalysis | ~87,600 h | Feasibility (cached in `cache/weather/`) |
| Runtime: Open-Meteo forecast | NWP | 168 h | Forecast |

**Real plants** (`training/plants.py`):

| Plant | Climate | kWp | Mount | Notes |
|---|---|---|---|---|
| Arbuckle CA (NREL 2107) | Csa | 893 | 25° south | 6+ years, also forecast training |
| Las Vegas NV (PVDAQ 34) | **BWh hot desert** | 146.6 | 11° south | 135 kW inverter, hW→W unit switch fixed |
| Las Vegas NV (PVDAQ 35) | BWh | 121.7 | ~6° **west** | Tilt fitted from POA sensor |
| Henderson NV (PVDAQ 1423) | BWh | **6.48** | 35° south | Nameplate corrected from metadata module count (record said 6.0) |
| Golden CO (PVDAQ 10) | BSk | 1.12 | 40° south | 2006 CIS thin film; ages at 1 %/yr |

The cleaning rules, rejected sites and licences are in [`data/README.md`](data/README.md).

---

## 6. Models

### 6.1 Physics (pvlib 0.16)
- **Irradiance and geometry:** NREL SPA solar position; Perez transposition with albedo 0.2; physical IAM (beam) plus Marion diffuse IAM.
- **Temperature:** Faiman cell temperature (u0 = 25, u1 = 6.84).
- **DC power:** PVWatts DC with γ = −0.35 %/°C.
- **Losses:** `pvwatts_losses` chain — soiling 4, shading 1, mismatch 2, wiring 2, connections 0.5, LID 1.5, nameplate 1, availability 2 % (≈ 13.2 % combined).
- **Inverter:** 97.5 % efficiency with AC clipping.
- **Optional tracking:** single-axis with backtracking.

### 6.2 Real-plant calibration

| Model card | k (mean) | k spread | Per-plant k |
|---|---|---|---|
| Hindcast (ERA5) | **0.964** | 5.2 % | 0.921 / 0.936 / 1.051 / 0.942 / 0.970 |
| Forecast (NWP) | **0.940** | 4.3 % | 3 plants |

Physics with the Indian default losses is about 3.6 % optimistic against real plants, so P50 is multiplied by k.

### 6.3 ML uncertainty model
- **Model:** XGBoost `reg:quantileerror` (α = 0.1, 0.5, 0.9), 600 trees, depth 6.
- **Target:** age- and k-normalised output per kWp.
- **Features:**
  - physics AC and DC
  - POA, GHI, DNI, diffuse fraction, clear-sky index
  - solar elevation, cos AOI
  - air and cell temperature, wind, cloud cover, RH
  - AC limit

  There is no location or calendar feature.
- **Calibration:** Mondrian conformal prediction on leave-one-plant-out predictions. A separate scale is fitted per clear-sky-index bin, hourly and daily.

| Hourly clear-sky-index bin (forecast) | <0.3 | 0.3–0.6 | 0.6–0.8 | 0.8–0.9 | >0.9 |
|---|---|---|---|---|---|
| Scale | 3.85 | 2.05 | 2.00 | 1.75 | 3.35 |

---

## 7. Validation results

| Test | Result |
|---|---|
| Golden test: physics vs PVGIS, 36 cities, same system | Yield mean **+4.6 %**, MAE 5.4 %, range −3.9 … +11.5 %; monthly correlation median 0.968 |
| Irradiance agreement, two ERA5 chains | POA mean +0.4 %, σ **4.1 %** (ERA5 lower in the haze-prone north, higher in the south) |
| Hindcast, leave-one-plant-out, hourly nRMSE (% of capacity) | physics **10.9 %**, ML P50 11.1 % |
| Forecast, leave-one-plant-out, hourly nRMSE | physics **12.5 %**, ML P50 12.3 %, 24 h persistence 19.6 % → **36 % skill** |
| P10–P90 coverage on unseen plants | uncalibrated 50 % (forecast) / 54 % (hindcast) → calibrated **80.2 % hourly, 80.3 % daily** |
| Lead-time error growth (Arbuckle, previous runs) | day 0–2: 1.00 · day 3: 1.015 · day 4: 1.065 · day 5: 1.094 · day 6: 1.098 |
| Clear-forecast days, daily measured ÷ physics P10–P90 | Henderson (desert) **0.90–1.07**; Arbuckle 0.82–1.15; Golden 0.76–1.24 |

**Decision recorded:** the ML point forecast does not beat physics out of sample, so physics × k is the P50 and ML is used only for the bands.

---

## 8. Current state

### Working (verified headless with Streamlit `AppTest`, no exceptions)
- All five tabs, the tilt optimiser, city comparison, the atlas map click-to-load, and JSON report download.
- Feasibility for any lat/lon in about 3 s once weather is cached (10-year ERA5 fetch ~5–20 s the first time). Live forecast in under 1 s.
- **Jaipur reference** (1 MWp, DC/AC 1.25):
  - Yield: P50 1,539 kWh/kWp (P90 1,406); CUF (AC) 22.0 %; PR 0.74.
  - Verdict: **VIABLE**.
- **Atlas range:** Leh 1,915 → Guwahati 1,323 kWh/kWp. Strongest monsoon dip is Mangaluru (−31 %).
- 30 tests pass (29 offline + 1 live-API smoke test); ruff is clean.

### Not verified
- The AI Analyst against the live Claude API (no credentials in this environment). Its tools are unit-tested offline.
- The Docker image build and the GitHub Actions run (files written, not executed here).

### Git
- **Nothing is committed.** The working tree has the v2 code; `src/` and the v1 model files show as deleted.
- Recommended: review, then commit on a branch (e.g. `v2-physics-first`).
- Generated large files are gitignored. Committed artefacts total about 40 MB:
  - models (2 × ~6 MB)
  - `reports/cv_predictions.parquet` (5.7 MB)
  - `data/real_plants_hourly.csv` (24 MB)

---

## 9. How good is it now?

| Dimension | v1 | v2 | Why |
|---|---|---|---|
| Physical correctness | 2 | **4** | Industry-standard pvlib chain, golden-tested on 36 cities, real-plant calibrated |
| ML methodology | 2 | **4** | Real targets, LOPO CV, honest baselines, conformal calibration with measured coverage; ML used only where it adds value |
| Data engineering | 2 | **4** | One validated weather boundary, units/time zones tested, per-year clock alignment, nameplate and unit-switch fixes |
| Financial model | 1.5 | — | Removed; out of scope by design |
| Software engineering | 1.5 | **3.5** | Package, tests, CI, pins, Docker, Makefile; no type-checking or API service yet |
| UI/UX | 3 | **4** | Interactive charts, persistent results, atlas, validation transparency |
| India-specific validity | — | **2.5** | Biggest gap: no Indian plant measurements |
| **Overall** | **~2.5** | **~3.8 / 5** | A credible, defensible engineering tool. Not yet bankable for Indian projects until calibrated on Indian plants |

---

## 10. Remaining problems and how to tackle them

| # | Problem | Impact | How to tackle |
|---|---|---|---|
| R1 | **No Indian plant data** in calibration | k and the bands come from US plants; monsoon cloud, dust and haze behaviour are untested | Email the authors of the IEEE DataPort Karnataka 72 kWp dataset (DOI 10.21227/x0zh-0y72); register for DKASC Alice Springs (hot-arid analogue); approach SECI/NTPC/state nodal agencies or rooftop installers for anonymised meter data. Add plants via `training/plants.py`; everything downstream retrains with `make train` |
| R2 | Forecast bands conservative on clear dry-season days | Fewer false alarms, but less sharp than possible | Once Indian data exists, add a climate-class dimension to the Mondrian bins, or weight calibration by climate similarity |
| R3 | Only 3 forecast-training plants; lead growth from 1 plant | Band calibration has limited diversity | Pull more PVDAQ systems with 2021+ data (the systems list has 158) and fetch previous-runs data for them |
| R4 | ERA5 regional irradiance bias (north low, south high, ±5 %) | Site-level P50 bias of a few % | Add satellite irradiance (CAMS radiation service, or NSRDB-India where available) as a second source; blend and report the spread |
| R6 | No horizon/far-shading, bifacial, or snow models | Hill sites and bifacial plants under- or over-estimated | pvlib `shading`, `bifacial.infinite_sheds`; horizon from PVGIS `printhorizon` |
| R7 | AI Analyst untested live, sequential tool runs | Slow multi-site comparisons | Run once with a key; parallelise `compare_sites`; add prompt caching for the system prompt and tools |
| R8 | Streamlit runs everything in-process | Not multi-user scalable | FastAPI service around `solarsense/` and a job queue for 10-year fetches; the UI calls the API |

---

## 11. Roadmap

**🔴 Next (data credibility)**
1. Acquire at least one Indian plant dataset (R1), retrain, and compare k for India vs US.
2. Run the AI Analyst live, then add an evaluation set of analyst questions with tool-grounded answers.

**🟠 Important**
3. Satellite irradiance as a second source (R4).
4. Day-ahead schedules in India's DSM format from the P10/P50/P90 forecast.
5. FastAPI service, background jobs, and a Postgres cache of site results (R8).

**🟢 Nice to have**
6. Bifacial, horizon, snow and soiling-schedule models (R6).
7. PDF report export, a multi-site portfolio view, and rooftop mode (kW-scale systems, roof orientation presets).

**🚀 Standout**
8. **Site-screening heatmap** for a state: grid the ERA5 cells, run physics, overlay OSM substations, land use and slope, and rank sites.
9. **Forecast verification dashboard** for an operating plant: forecast vs metered output, rolling skill.
10. **Digital twin:** upload inverter CSVs to get measured-vs-expected tracking, soiling loss in kWh/day, and when cleaning recovers the most energy (the residual model already exists).
11. **Satellite nowcasting** (INSAT-3D / Himawari) for 0–4 h.
12. **Battery sizing** for self-consumption: shift midday surplus to an evening load profile (energy only).

---

## 12. How to run and rebuild

```bash
pip install -r requirements.txt
streamlit run app/app.py           # works immediately with the committed models

make test                          # 29 offline tests
make data                          # rebuild datasets (network, cached)
make train                         # training set → CV → models → lead time
make validate                      # golden test vs PVGIS
make atlas                         # India atlas
```

Order matters inside `make data`: `build_real_plant_dataset.py` writes `real_plants_hourly.csv` fresh, then `build_pvdaq_dataset.py` merges the PVDAQ plants into it.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| GHI / DNI / DHI | Global horizontal / direct normal / diffuse horizontal irradiance (W/m²) |
| POA | Plane-of-array irradiance on the tilted module |
| kWp | DC nameplate of the modules; AC rating = kWp ÷ DC/AC ratio |
| CUF | Capacity utilisation factor on AC rating: energy ÷ (AC kW × 8,760). The Indian convention |
| Specific yield | kWh per kWp per year |
| PR | Performance ratio: specific yield ÷ POA irradiation |
| P50 / P90 | Energy expected to be exceeded with 50 % / 90 % probability |
| k | Measured ÷ physics energy on a real plant (age- and nameplate-corrected) |
| Leave-one-plant-out | Train on all plants but one, test on the held-out plant |
| Conformal prediction | Scales prediction intervals so they reach a target coverage on held-out data |
| Mondrian conformal | Conformal calibration done separately per group (here, clear-sky-index bins) |
| Clear-sky index | GHI ÷ clear-sky GHI: 1 = cloudless, about 0.2 = overcast |
| ERA5 | ECMWF global reanalysis (weather model fitted to observations) |
| PVDAQ | NREL's public archive of measured PV system data |
| DSM | India's Deviation Settlement Mechanism (penalises forecast error) |
