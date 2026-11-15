# ☀️ SolarSense

**Physics-first, real-plant-calibrated solar energy yield assessment and probabilistic forecasting for India.**

`pvlib` · `XGBoost quantile regression` · `conformal prediction` · `ERA5 / Open-Meteo` · `NREL PVDAQ` · `Streamlit` · `Claude tool use`

SolarSense answers three questions for any location:

| Question | Answer you get |
|---|---|
| **How much energy will this site produce?** | P50 / P90 annual energy from 10 years of hourly weather, specific yield, CUF, performance ratio, loss waterfall, monthly profile, uncertainty budget, and a VIABLE / MARGINAL / NOT VIABLE energy verdict with reasons |
| **What will my plant produce this week?** | Hourly and daily P10 / P50 / P90 forecast with uncertainty bands calibrated on real plants |
| **Where in India is best?** | A 36-city atlas: yield ranking, monsoon dip, optimal tilt, side-by-side comparison |

An optional **AI Analyst** (Claude) answers questions like *"Which yields more for 5 MW, Bikaner or Coimbatore?"* by calling SolarSense's own engines. It may only quote numbers those tools return.

---

## How it works

```
                 ┌──────────────── weather (UTC, W/m², validated in one place) ────────────────┐
                 │  ERA5 reanalysis via Open-Meteo (10 yr)  ·  NASA POWER fallback  ·  NWP forecast │
                 └──────────────────────────────────────┬──────────────────────────────────────┘
                                                        ▼
  PHYSICS (pvlib)   solar position → Perez POA → IAM → Faiman cell temp → PVWatts DC → loss chain → inverter + clipping
                                                        ▼
  REAL-PLANT        × k = measured ÷ physics on 5 metered NREL plants (age- and nameplate-corrected)
  CALIBRATION       spread of k across plants = energy uncertainty for a new plant
                                                        ▼
  UNCERTAINTY       quantile XGBoost half-widths × Mondrian conformal scale (per clear-sky-index bin)
  (ML)              × measured error growth by forecast lead day
                                                        ▼
  OUTPUTS           P50/P90 energy · loss waterfall · uncertainty budget · verdict · P10–P90 forecast
```

**Why physics first?** I trained ML models on 99,909 real daylight hours to predict output directly. Out of sample they were **no better than pvlib** (−1.9 % to +1.6 % hourly nRMSE). So physics gives the central estimate, and the ML does what physics can't: calibrated uncertainty.

## Validation (all out-of-sample)

| Test | Result |
|---|---|
| Physics vs PVGIS, 36 Indian cities (same system, no losses) | mean **+4.6 %**, MAE 5.4 %, monthly-shape correlation 0.97 |
| Weather-source spread (two ERA5 processing chains, plane-of-array) | **4.1 %** (1σ), used in P90 |
| Physics vs **real metered output**, 5 US plants (BWh desert, Csa, BSk) | measured ÷ physics **k = 0.964 ± 5.2 %**, now applied as a calibration |
| 7-day forecast vs 24 h persistence (leave-one-plant-out) | **36 % lower error** (hourly nRMSE 12.5 % of capacity) |
| P10–P90 band coverage on unseen plants | raw ML 50–54 % → **80 % after conformal calibration** (hourly *and* daily) |
| Forecast error growth with lead time (Open-Meteo previous runs) | day 6 is **+10 %** vs day 0 (floor of +5 %/day applied for cloudier climates) |

Reports live in [`reports/`](reports/) and in the app's **Model & Validation** tab. The model cards are [`models/*_model_card.json`](models/).

**Example: Jaipur, 1 MWp fixed tilt, DC/AC 1.25.** P50 is 1,539 kWh/kWp/yr and P90 is 1,406. CUF (AC) is 22.0 % and performance ratio 0.74. Verdict: **VIABLE** (P50 ≥ 1,500 kWh/kWp).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The trained models, model cards, reports, atlas and real-plant dataset are committed, so the app works straight away. Weather is fetched on demand and cached in `cache/`.

**Rebuild everything from public data:**

```bash
make data       # NREL real plants + India ERA5 archive + PVGIS reference (~20 min, cached)
make train      # training set → leave-one-plant-out CV → models + cards → lead-time analysis
make validate   # golden test vs PVGIS on 36 cities
make atlas      # India atlas
make test       # 29 offline tests (pytest -m network for live API smoke tests)
```

**Docker:** `docker build -t solarsense . && docker run -p 8501:8501 solarsense`

**AI Analyst (optional):** `pip install anthropic`, then `export ANTHROPIC_API_KEY=...` (or `ant auth login`). It uses `claude-opus-5-5` with server-side refusal fallback.

## Project layout

```
solarsense/          core package
  config.py          paths, physical defaults, verdict thresholds, uncertainty constants
  weather.py         Open-Meteo (ERA5, forecast, historical forecast), NASA POWER; WeatherFrame contract
  physics.py         pvlib simulation, loss waterfall, tilt optimiser
  ml.py              feature contract, quantile XGBoost wrapper, metrics
  calibration.py     real-plant k, conformal band scales, lead-time factors (from model cards)
  feasibility.py     P50/P90, uncertainty budget, monthly profile, verdict
  forecast.py        7-day P10/P50/P90
  agent.py           Claude tool-use analyst
training/            real-plant training set, LOPO CV + conformal calibration, lead-time study
scripts/             data builders (NREL PVDAQ, Open-Meteo archive, PVGIS), golden test, atlas
app/                 Streamlit UI
tests/               offline unit + pipeline tests, network smoke tests
data/ models/ reports/
```

See [`data/README.md`](data/README.md) for every dataset, its source and licence, and [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) for the full technical write-up.

## Limitations (honest list)

- **No Indian plant measurements yet.** Calibration uses 5 US plants (3 for forecasting). The Nevada desert sites are good analogues for Rajasthan, but monsoon-cloud behaviour is not represented. The only multi-year Indian plant dataset found (IEEE DataPort, Karnataka 72 kWp) is paywalled.
- **Forecast bands are likely conservative for clear dry-season Indian days.** They are calibrated on plants that include Colorado convection and California tule fog.
- Archived forecasts used for training are mostly short-lead. Lead-time growth was measured at one plant.
- Energy only by design: no costs, tariffs or returns. The verdict is based on specific yield alone.

---

Developed by **Kanakamurthy H**, IIT Bombay · Data: Open-Meteo (ERA5 © ECMWF, CC-BY 4.0), NASA POWER, NREL PVDAQ (CC-BY 4.0), PVGIS (© EU) · Physics: [pvlib](https://pvlib-python.readthedocs.io)
