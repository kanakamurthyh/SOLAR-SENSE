# Data sources

All timestamps are **UTC**. Irradiance in the generated CSVs is **kW/m²** (Open-Meteo raw files are W/m²). Power is **kW**.

| File | What | Rows | Real or simulated | Regenerate with |
|---|---|---|---|---|
| `real_plants_hourly.csv` | Metered AC output + on-site irradiance/temperature from 5 US PV systems, 2015–2024, hourly | 141 891 | **Real** | `scripts/build_real_plant_dataset.py` then `scripts/build_pvdaq_dataset.py` |
| `training_set.parquet` | Real-plant targets joined with ERA5 (hindcast) and archived NWP (forecast) weather + physics features | 99 909 daylight h | Real targets | `python3 -m training.build_training_set` (gitignored) |
| `india_atlas.csv` | Calibrated yield, CUF, monsoon dip, optimal tilt for the 36 cities | 36 | Derived | `scripts/build_india_atlas.py` |
| `india_openmeteo_hourly_2020_2022.csv` | ERA5-based hourly weather (GHI, **DNI, DHI**, temp, RH, wind, cloud) for 36 Indian cities | 946 944 | Reanalysis (weather only, no PV output) | `scripts/fetch_openmeteo_india_archive.py` |
| `india_pvgis_hourly.csv` | PVGIS simulated 1 kWp PV output + POA irradiance, 36 cities, 2020–2022 (PVGIS uses its **ERA5** radiation database for India) | 946 944 | **Simulated** | `scripts/build_india_pvgis_dataset.py` |
| `india_locations.csv` | The 36-site list | 36 | — | hand-curated |
| `Plant_1/2_*.csv` | Kaggle two-plant India dataset, 15-min, 34 days (May–Jun 2020). Not used by the models: the plant locations are unknown, so no weather can be matched | ~136 k raw | Real, short | Kaggle |

The two large India files are gitignored (68 MB and 108 MB); rerun the scripts to rebuild them.

## `real_plants_hourly.csv` — plants

| LOCATION_ID | Source | Location / climate | kWp behind meter | Mount | Irradiance | Coverage (full years) |
|---|---|---|---|---|---|---|
| `nrel_2107_arbuckle_ca` | NREL 2023 Solar Data Prize, site 2107 | Arbuckle CA, Csa | 893 (nameplate) | fixed 25°/180° | POA | Dec 2017 – Nov 2024 (2018–2023) |
| `pvdaq_34_las_vegas_nv` | NREL PVDAQ system 34 | Las Vegas NV, **BWh hot desert** | 146.6 | fixed 11°/180° | POA | 2015 – Jul 2020 (2015–2017 clean; 2018–19 gappy) |
| `pvdaq_35_las_vegas_nv` | NREL PVDAQ system 35 | Las Vegas NV, **BWh**, gym roof | 121.7 | fixed, az 270° (west-facing) | POA | 2015 – Jul 2020 (2016–2017 clean) |
| `pvdaq_1423_henderson_nv` | NREL PVDAQ system 1423 | Henderson NV, **BWh** | 6.0 in this file (system record); training uses **6.48** = 24 × 270 W modules from the same metadata | fixed 35°/180° | POA | Jun 2021 – Dec 2022 (2022 full; public bucket is sparse before that) |
| `pvdaq_10_golden_co` | NREL PVDAQ system 10 | Golden CO, BSk cold semi-arid | 1.12 | fixed 40°/180° | POA | Jan 2020 – Feb 2023 (2020–2022) |

Columns: `DATE_TIME, AC_POWER, AC_POWER_PER_KWP, IRRADIATION, IRRADIANCE_TYPE, DNI, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, WIND_SPEED_10M, LATITUDE, LONGITUDE, LOCATION_ID, SYSTEM_CAPACITY_KW, CAPACITY_SOURCE, MOUNT`.
`DNI`/`MODULE_TEMPERATURE`/`WIND_SPEED_10M` are present only where the site measures them.

### Cleaning applied
* Source timestamps are in a fixed local offset that can change between years (system 34/35 switch from UTC−8 to UTC−7 in 2019); the offset is inferred **per calendar year** from the daylight window vs solar noon and removed (±1 h). Months whose irradiance peak still sits >2 h from solar noon (logger clock faults) are dropped.
* PVDAQ loggers switch power units between years (system 34: hectowatts → watts in 2018). Per calendar year the power-of-ten scale that puts P99.5 near 0.8 × nameplate is applied.
* Meter spikes above 1.25 × the sustained peak and negative values are removed; 15-min / 1-min samples are averaged to hourly.
* Calendar months whose daytime output/irradiance ratio falls below 25 % of the site median (outages, logger faults) are dropped.
* Yearly specific yields after cleaning: Arbuckle 1 300–1 540, Las Vegas 1 370–1 590 (clean years), Henderson 1 920 (2022), Golden 1 130–1 440 kWh/kWp — all in the physically expected range.

### Sites examined and rejected
* NREL prize **2105 (Maui)**: the public `meter_ac_output` channel is flat ~5.6 kW day and night — not PV output. `--sites 2105` re-enables it.
* NREL prize **9068 (Kersey CO)**: 35 % negative readings and whole months of reversed output. `--sites 9068` re-enables it.
* NREL prize **7333 / 9069**: thousands of raw string-level SCADA files, no compact meter file.

## Licences / attribution
* NREL PVDAQ and 2023 Solar Data Prize: public domain / CC-BY-4.0, https://data.openei.org/submissions/4568 — cite NREL PVDAQ.
* Open-Meteo: CC-BY-4.0, https://open-meteo.com (ERA5 © ECMWF).
* PVGIS: © European Union, free for reuse with attribution.

## Other real-plant datasets worth adding (need registration, not automated here)
* **DKASC Alice Springs** (Australia, 5-min, 15+ years, dozens of technologies, hot arid — very India-like): free after a name/email form at https://dkasolarcentre.com.au/download?location=alice-springs
* **PVOD** (10 plants, Hebei China, 15-min, 1 year, includes NWP forecasts — ideal for forecast-correction ML): https://github.com/yaotc/PVODataset (data via Science Data Bank, DOI 10.11922/sciencedb.01094)
* **72 kWp plant, Karnataka, hourly, May 2021 – Sep 2024** (the only multi-year Indian plant found): IEEE DataPort, DOI 10.21227/x0zh-0y72 — requires an IEEE DataPort subscription; user comments say the file is sometimes missing.
