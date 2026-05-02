# SolarSense

**Solar Plant Feasibility Analysis & Power Forecasting**

`Python` · `XGBoost` · `Streamlit` · `Open-Meteo API` · `NASA POWER API`

---

## Overview

SolarSense is a machine learning-powered decision support tool that answers two fundamental questions for anyone evaluating a solar investment:

- **Is this location viable?** Historical climate analysis using 10 years of NASA POWER satellite data to estimate annual energy output, capacity factor, and financial payback period.
- **How much will it produce this week?** Real-time 7-day operational forecast using Open-Meteo hourly weather data fed through the trained ML model.

The tool works for any location on Earth, supports any plant capacity, and delivers a **GO / MARGINAL / NOT VIABLE** verdict with full financial projection including payback period and 25-year P&L.

---

## Model Performance

Trained on two Indian utility-scale solar plants (34 days, 3,507 rows after inverter aggregation).

| Metric | XGBoost | Linear Regression (baseline) |
|---|---|---|
| R² Score | **0.9257** | 0.8728 |
| nRMSE | **6.62%** | 8.66% |
| Skill Score | **67.4%** | 57.3% |

> **Key finding:** Inverter aggregation was the critical preprocessing fix. Aggregating 22 inverters per timestamp from per-inverter (~75 kW) to whole-plant (~29,000 kW) eliminated data inconsistency and lifted R² from 0.76 to 0.93.

---

## Project Structure

```
SolarSense/
├── data/
│   ├── Plant_1_Generation_Data.csv
│   ├── Plant_1_Weather_Sensor_Data.csv
│   ├── Plant_2_Generation_Data.csv
│   ├── Plant_2_Weather_Sensor_Data.csv
│   └── merged_clean.csv                  ← auto-generated on first run
├── src/
│   ├── config.py                         ← all constants and settings
│   ├── model.py                          ← train, save, load, predict
│   ├── api.py                            ← Open-Meteo + NASA POWER calls
│   ├── feasibility.py                    ← GO/NO-GO decision engine
│   ├── evaluate.py                       ← metrics and chart functions
│   ├── test.py                           ← terminal test for any location
│   └── utils/
│       ├── preprocessing.py              ← full data pipeline
│       └── eda.ipynb                     ← exploratory data analysis
├── models/
│   ├── xgboost_model.pkl                 ← saved after training
│   └── metadata.json
├── app/
│   └── app.py                            ← Streamlit application
└── requirements.txt
```

---

## What Each File Does

### `src/config.py`
Single source of truth for all constants — file paths, feature list, model hyperparameters, API URLs, and feasibility thresholds. Every other file imports from here. Nothing is hardcoded elsewhere.

### `src/utils/preprocessing.py`
Full data pipeline in one place. Loads both Kaggle plant CSVs, aggregates all inverters per timestamp into whole-plant totals, cleans data (drops nighttime rows, negative power, outliers), engineers all 10 features, and applies a date-based 80/20 split. Saves `merged_clean.csv` on first run so subsequent runs skip reprocessing. The `prepare_inference_data()` function applies identical feature engineering to Open-Meteo forecast data, guaranteeing training and inference always use the same feature computation.

### `src/model.py`
Trains Linear Regression (baseline) and XGBoost with early stopping, compares R², and saves the better model to `models/`. Exposes `load_model()` and `predict()` for inference.

### `src/api.py`
All external API calls in one place. `get_forecast()` fetches 7-day hourly weather from Open-Meteo. `get_historical()` fetches 10-year monthly climate averages from the NASA POWER climatology endpoint. Both are free with no API key required.

### `src/feasibility.py`
The core decision engine. `run_feasibility()` builds synthetic typical days from NASA POWER GHI data, converts daily energy (kWh/m²/day) to hourly instantaneous irradiance using a bell-curve solar model, runs model predictions for all 12 months, computes annual output and capacity factor, and returns a GO/MARGINAL/NOT VIABLE verdict with financial breakdown. `run_forecast()` handles 7-day operational prediction.

### `src/evaluate.py`
Computes MAE, RMSE, nRMSE, R², and skill score vs a persistence baseline. All chart functions return matplotlib Figure objects that Streamlit renders directly.

### `src/test.py`
Terminal test script. Edit `LAT`, `LON`, `CAPACITY_KW`, `COST_INR`, `TARIFF` at the top and run without the UI. Prints 7-day forecast and monthly output table directly to terminal.

### `app/app.py`
Streamlit application with three tabs. **Feasibility Analysis** shows monthly output chart, month-by-month detail table, verdict banner, 25-year P&L projection, and financial summary. **7-Day Forecast** shows daily bar chart, hourly curve for the best day, and a day-by-day detail table. **Model Diagnostics** shows performance metrics, feature importance chart, and model comparison table. All numerical output uses IBM Plex Mono for readability.

---

## Engineered Features

All 10 features derived from 3 raw sensor inputs and the timestamp:

| Feature | Source | Description |
|---|---|---|
| `IRRADIATION` | Raw sensor | Instantaneous solar irradiance — strongest predictor |
| `AMBIENT_TEMPERATURE` | Raw sensor | Air temperature at 2m |
| `MODULE_TEMPERATURE` | Raw / estimated | Panel surface temp; estimated via Sandia model at inference |
| `temp_delta` | Derived | Panel temp minus air temp; high delta signals efficiency loss |
| `irradiation_sq` | Derived | Captures non-linear curve in irradiance vs output |
| `irr_x_temp` | Derived | Interaction term between irradiance and temperature |
| `hour_sin`, `hour_cos` | Time | Cyclic hour encoding; hour 23 and hour 0 are adjacent |
| `month_sin`, `month_cos` | Time | Cyclic seasonal encoding |
| `day_of_year` | Time | Captures gradual seasonal sun-path shift |
| `PLANT_ID` | Metadata | Per-plant differences in tilt, shading, inverter efficiency |

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download dataset** — place all 4 CSV files in `data/`
```
https://www.kaggle.com/datasets/anikannal/solar-power-generation-data
```

**3. Train the model**
```bash
cd src
python model.py
```

**4. Launch the app**
```bash
streamlit run app/app.py
```

**5. Test any location from terminal**
```bash
cd src
python test.py
```

---

## How the Pipeline Works

**Feasibility Analysis (Historical)**
1. Fetch 10-year monthly GHI and temperature averages from NASA POWER
2. Convert daily GHI (kWh/m²/day) to hourly instantaneous irradiance: `peak_irr = (π/2) × GHI / daylight_hours`
3. Run trained model on synthetic typical days for all 12 months
4. Aggregate to monthly kWh → annual output → capacity factor → payback period
5. Verdict threshold: capacity factor ≥ 18% and payback ≤ 10 years → VIABLE

**Operational Forecast (7-Day)**
1. Fetch hourly weather from Open-Meteo for the target location
2. Apply identical feature engineering as training
3. Scale predictions by capacity ratio (user plant / training plant)
4. Output daily totals, hourly curve, and day-by-day breakdown

---

## Dataset Note

Training covers May–June 2020 only (34 days, 2 Indian plants). A full-year dataset would push R² from 0.93 to ~0.97 by adding seasonal variation. Module temperature is estimated at inference via the Sandia simplified model since Open-Meteo does not provide panel surface temperature.

---

## Requirements

```
pandas · numpy · scikit-learn · xgboost · joblib · matplotlib · requests · streamlit
```

---
DEVELOPED BY KANAKAMURTHY H, Kanakamurthyh@gmail.com, 24B2151@iitb.ac.in, IIT BOMBAY. 
