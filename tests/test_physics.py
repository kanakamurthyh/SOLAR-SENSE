import numpy as np
import pytest

from solarsense.physics import PVSystem, loss_waterfall, optimise_orientation, simulate

from .conftest import JAIPUR


def test_zero_at_night_and_bounded_by_ac_rating(jaipur_week):
    s = PVSystem(capacity_kwp=100, dc_ac_ratio=1.25)
    sim = simulate(jaipur_week, s)
    assert (sim.loc[sim["solar_zenith"] >= 90, "ac"] == 0).all()
    assert sim["ac"].max() <= s.ac_capacity_kw + 1e-9
    assert (sim["ac"] >= 0).all()


def test_peak_power_near_solar_noon_utc(jaipur_week):
    """Guards the v1 timezone bug: peak must sit at solar noon in UTC (~07:00 for Jaipur)."""
    sim = simulate(jaipur_week, PVSystem(capacity_kwp=1, dc_ac_ratio=1.0))
    peak_hour = sim["ac"].groupby(sim.index.hour).mean().idxmax()
    solar_noon_utc = 12 - JAIPUR[1] / 15
    assert abs(peak_hour + 0.5 - solar_noon_utc) <= 1.0


def test_clear_day_specific_yield_is_physical(jaipur_week):
    sim = simulate(jaipur_week, PVSystem(capacity_kwp=1))
    daily = sim["ac"].resample("D").sum().iloc[1:-1]
    assert daily.between(4.5, 8.0).all(), daily


def test_system_losses_applied_exactly(jaipur_week):
    s = PVSystem(capacity_kwp=10)
    sim = simulate(jaipur_week, s)
    day = sim["dc_gross"] > 0
    ratio = (sim.loc[day, "dc_net"] / sim.loc[day, "dc_gross"]).unique()
    assert np.allclose(ratio, 1 - s.system_loss_fraction)
    assert 0.10 < s.system_loss_fraction < 0.16


def test_high_dc_ac_ratio_clips(jaipur_week):
    sim = simulate(jaipur_week, PVSystem(capacity_kwp=100, dc_ac_ratio=1.6))
    assert sim["clipped"].sum() > 0


def test_default_tilt_and_azimuth():
    s = PVSystem()
    assert s.resolved_tilt(26.9) == pytest.approx(24.2)
    assert s.resolved_azimuth(26.9) == 180.0
    assert s.resolved_azimuth(-10) == 0.0


def test_tracker_beats_fixed_on_clear_sky(jaipur_week):
    fixed = simulate(jaipur_week, PVSystem(capacity_kwp=1, dc_ac_ratio=1.0))["ac"].sum()
    tracker = simulate(jaipur_week, PVSystem(capacity_kwp=1, dc_ac_ratio=1.0, tracking="single_axis"))["ac"].sum()
    assert tracker > fixed * 1.1


def test_waterfall_ends_at_delivered_energy(jaipur_year):
    s = PVSystem(capacity_kwp=1)
    sim = simulate(jaipur_year, s)
    wf = loss_waterfall(sim, jaipur_year, s)
    assert wf["kwh_per_kwp"].iloc[-1] == pytest.approx(sim["ac"].sum() / (len(sim) / 8766), rel=1e-9)
    assert wf["kwh_per_kwp"].iloc[1] > wf["kwh_per_kwp"].iloc[0]      # tilt gain
    assert (wf["kwh_per_kwp"].diff().iloc[2:] <= 1e-9).all()          # every later step is a loss


def test_orientation_optimum_near_latitude(jaipur_year):
    best = optimise_orientation(jaipur_year, PVSystem(capacity_kwp=1), *JAIPUR, tilts=range(0, 46, 5)).iloc[0]
    assert 15 <= best["tilt"] <= 40


def test_invalid_system_rejected():
    with pytest.raises(ValueError):
        PVSystem(capacity_kwp=0)
    with pytest.raises(ValueError):
        PVSystem(dc_ac_ratio=3.0)
