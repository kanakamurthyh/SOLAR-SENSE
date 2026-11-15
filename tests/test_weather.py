import pandas as pd
import pytest

from solarsense.weather import _openmeteo_frame, complete_years, validate_weather


def test_valid_frame_passes(jaipur_week):
    assert validate_weather(jaipur_week) is jaipur_week


def test_rejects_kw_per_m2(jaipur_week):
    """The v1 forecast bug: irradiance in the wrong unit must fail loudly."""
    bad = jaipur_week.copy()
    bad[["ghi", "dni", "dhi"]] /= 1000.0
    with pytest.raises(ValueError, match="kW/m²"):
        validate_weather(bad)


def test_rejects_naive_and_non_utc(jaipur_week):
    with pytest.raises(ValueError, match="tz-aware"):
        validate_weather(jaipur_week.tz_localize(None))
    with pytest.raises(ValueError, match="UTC"):
        validate_weather(jaipur_week.tz_convert("Asia/Kolkata"))


def test_rejects_wind_in_kmh(jaipur_week):
    bad = jaipur_week.copy()
    bad["wind_speed"] = 120.0
    with pytest.raises(ValueError, match="wind_speed"):
        validate_weather(bad)


def test_openmeteo_parsing_shifts_to_interval_centre_and_keeps_units():
    payload = {
        "elevation": 431.0,
        "hourly": {
            "time": ["2024-03-01T06:00", "2024-03-01T07:00"],
            "shortwave_radiation": [600.0, 700.0], "direct_normal_irradiance": [800.0, 850.0],
            "diffuse_radiation": [100.0, 110.0], "temperature_2m": [25.0, 27.0],
            "wind_speed_10m": [3.0, 3.5], "relative_humidity_2m": [30, 28], "cloud_cover": [0, 5],
        },
    }
    df = _openmeteo_frame(payload, "test", 26.9, 75.8)
    # value labelled 07:00 is the 06:00-07:00 mean -> centre 06:30 UTC
    assert df.index[1] == pd.Timestamp("2024-03-01 06:30", tz="UTC")
    assert df["ghi"].tolist() == [600.0, 700.0]          # W/m², not converted
    assert df.attrs["elevation"] == 431.0


def test_complete_years(jaipur_year):
    assert complete_years(jaipur_year) == [2023]
    assert complete_years(jaipur_year.iloc[:5000]) == []
