"""Live API smoke tests. Run with: pytest -m network"""
import pytest

from solarsense.weather import fetch_openmeteo_forecast

pytestmark = pytest.mark.network


def test_live_forecast_contract():
    w = fetch_openmeteo_forecast(26.9124, 75.7873, days=2)
    assert str(w.index.tz) == "UTC"
    assert 300 < w["ghi"].max() < 1300          # W/m², daytime Jaipur
    assert w.attrs["source"] == "open-meteo-forecast"
