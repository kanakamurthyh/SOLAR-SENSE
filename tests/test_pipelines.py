"""End-to-end feasibility and forecast on synthetic weather (no network)."""
import pandas as pd
import pytest

from solarsense.calibration import Calibration
from solarsense.feasibility import make_verdict, run_feasibility
from solarsense.forecast import local_timezone, run_forecast
from solarsense.physics import PVSystem

from .conftest import JAIPUR, make_clear_weather


@pytest.fixture
def two_years():
    a = make_clear_weather(*JAIPUR, start="2022-01-01", days=365)
    b = make_clear_weather(*JAIPUR, start="2023-01-01", days=365, cloud_factor=0.9)
    w = pd.concat([a, b])
    w.attrs = a.attrs
    return w


def test_feasibility_end_to_end(two_years):
    r = run_feasibility(*JAIPUR, PVSystem(capacity_kwp=1000), weather=two_years)
    assert r.years == [2022, 2023]
    assert r.p90_kwh < r.p50_kwh < r.annual_by_year.max() * 1.0001
    assert 1300 < r.specific_yield_p50 < 2300
    assert len(r.monthly) == 12
    assert r.waterfall["step"].iloc[-1].startswith("Real-plant calibration")
    assert r.verdict["label"] in {"VIABLE", "MARGINAL", "NOT VIABLE"}
    assert not hasattr(r, "finance_p50")


def test_energy_only_verdict_thresholds():
    assert make_verdict(1600, 1500)["label"] == "VIABLE"
    assert make_verdict(1400, 1300)["label"] == "MARGINAL"
    assert make_verdict(1200, 1100)["label"] == "NOT VIABLE"
    assert any("Caution" in r for r in make_verdict(1520, 1250)["reasons"])


def test_forecast_units_bands_and_local_time(jaipur_week):
    """Regression for the v1 bug (2.4x overestimate): daily kWh/kWp must be physical."""
    fc = run_forecast(*JAIPUR, PVSystem(capacity_kwp=1000), weather=jaipur_week)
    h, d = fc.hourly, fc.daily
    assert (h["p10_kw"] <= h["p50_kw"] + 1e-9).all() and (h["p50_kw"] <= h["p90_kw"] + 1e-9).all()
    assert h["p90_kw"].max() <= fc.ac_capacity_kw + 1e-6
    assert d["p50_kwh"].between(3_500, 8_000).all(), d
    assert str(h["local_time"].dt.tz) == "Asia/Kolkata"


def test_local_timezone():
    assert local_timezone(*JAIPUR) == "Asia/Kolkata"
    assert str(local_timezone(51.5, 0.0)) == "UTC"


def test_calibration_lead_and_bins():
    c = Calibration(available=True, band_scale=2.0, hourly_bins=(0.0, 0.5, 1.6), hourly_scales={"0": 3.0, "1": 1.0},
                    lead_ratio={"0": 1.0, "6": 1.1})
    assert c.lead_factor(0) == 1.0
    assert c.lead_factor(6) == pytest.approx(1.3)   # floor 1 + 0.05*6 beats measured 1.1
    assert list(c.hourly_scale(pd.Series([0.2, 0.9, None]))) == [3.0, 1.0, 3.0]
