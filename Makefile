# SolarSense pipeline. Steps must run in this order; each is cached and safe to re-run.
PY ?= python3

.PHONY: install app test lint data train validate atlas all

install:            ## install runtime + dev dependencies
	$(PY) -m pip install -r requirements.txt pytest ruff

app:                ## launch the Streamlit app
	streamlit run app/app.py

test:               ## offline unit tests (no network)
	$(PY) -m pytest -m "not network" -q

lint:
	ruff check solarsense training scripts app tests

data:               ## real plants (NREL) + India weather archive + PVGIS reference
	$(PY) scripts/build_real_plant_dataset.py      # writes data/real_plants_hourly.csv fresh (site 2107)
	$(PY) scripts/build_pvdaq_dataset.py           # MERGES PVDAQ plants into it - must run after the line above
	$(PY) scripts/fetch_openmeteo_india_archive.py
	$(PY) scripts/build_india_pvgis_dataset.py

train:              ## training set from real plants -> models + model cards + reports
	$(PY) -m training.build_training_set
	$(PY) -m training.train
	$(PY) -m training.lead_time

validate:           ## golden test: physics vs PVGIS on 36 Indian cities
	$(PY) scripts/validate_physics_vs_pvgis.py

atlas:              ## 36-city India atlas (needs trained calibration)
	$(PY) scripts/build_india_atlas.py

all: data train validate atlas test
