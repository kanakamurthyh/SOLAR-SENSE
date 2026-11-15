"""
Site feasibility: long-term energy yield (P50 / P90), losses, verdict.

    10 years of hourly ERA5 weather (NASA POWER fallback)
      -> physics with the user's plant and loss chain
      -> x real-plant calibration k (hindcast card)
      -> annual energy per year -> P50 and year-to-year variability
      -> P90 from combined uncertainty:
           sigma_model  plant-to-plant spread on real plants (k_std)
           sigma_source weather-source spread over 36 Indian cities
           sigma_iav    inter-annual variability at this site
      -> verdict on specific yield
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from solarsense.calibration import load_calibration
from solarsense.config import (
    HISTORY_END_YEAR,
    HISTORY_START_YEAR,
    SOURCE_UNCERTAINTY_PCT,
    SPECIFIC_YIELD_MARGINAL,
    SPECIFIC_YIELD_VIABLE,
)
from solarsense.physics import PVSystem, loss_waterfall, optimise_orientation, simulate
from solarsense.weather import complete_years, get_history

Z90 = 1.2816  # one-sided 90 % normal quantile

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass
class FeasibilityResult:
    latitude: float
    longitude: float
    system: PVSystem
    weather_source: str
    years: list
    annual_by_year: pd.Series
    p50_kwh: float
    p90_kwh: float              # long-term (10-year average) P90
    p90_1y_kwh: float           # single-year P90
    uncertainty: dict
    monthly: pd.DataFrame
    waterfall: pd.DataFrame
    climate: dict
    tilt: float
    azimuth: float
    verdict: dict = field(default_factory=dict)
    calibrated: bool = False

    @property
    def specific_yield_p50(self) -> float:
        return self.p50_kwh / self.system.capacity_kwp

    @property
    def specific_yield_p90(self) -> float:
        return self.p90_kwh / self.system.capacity_kwp

    @property
    def cuf_ac_pct(self) -> float:
        """Capacity utilisation factor on AC rating - the Indian industry convention."""
        return self.p50_kwh / (self.system.ac_capacity_kw * 8760) * 100

    @property
    def cf_dc_pct(self) -> float:
        return self.p50_kwh / (self.system.capacity_kwp * 8760) * 100


def make_verdict(sy_p50: float, sy_p90: float) -> dict:
    """VIABLE / MARGINAL / NOT VIABLE from P50 specific yield, with P90 as context."""
    if sy_p50 >= SPECIFIC_YIELD_VIABLE:
        label = "VIABLE"
    elif sy_p50 >= SPECIFIC_YIELD_MARGINAL:
        label = "MARGINAL"
    else:
        label = "NOT VIABLE"
    reasons = [
        f"P50 specific yield {sy_p50:,.0f} kWh/kWp/yr "
        f"(viable ≥ {SPECIFIC_YIELD_VIABLE:,.0f}, marginal ≥ {SPECIFIC_YIELD_MARGINAL:,.0f}).",
        f"P90 specific yield {sy_p90:,.0f} kWh/kWp/yr - exceeded in 9 years out of 10 on a long-term average.",
    ]
    if label != "NOT VIABLE" and sy_p90 < SPECIFIC_YIELD_MARGINAL:
        reasons.append("Caution: the P90 case falls below the marginal threshold.")
    return {"label": label, "reasons": reasons}


def _climate_summary(weather: pd.DataFrame, sim: pd.DataFrame, monthly_p50: pd.Series) -> dict:
    day = sim["solar_elevation"] > 10
    csi = sim.loc[day, "clearsky_index"]
    monsoon = [6, 7, 8, 9]
    m = monthly_p50
    dry = m.drop(index=monsoon).mean()
    return {
        "ghi_kwh_m2_yr": float(weather["ghi"].sum() / 1000 / (len(weather) / 8766)),
        "mean_clearsky_index": float(csi.mean()),
        "monsoon_clearsky_index": float(csi[csi.index.month.isin(monsoon)].mean()),
        "monsoon_dip_pct": float((m.loc[monsoon].mean() / dry - 1) * 100) if dry > 0 else 0.0,
        "mean_daytime_temp_c": float(weather.loc[day, "temp_air"].mean()),
        "mean_cell_temp_c": float(sim.loc[day, "temp_cell"].mean()),
        "best_month": MONTHS[int(m.idxmax()) - 1],
        "worst_month": MONTHS[int(m.idxmin()) - 1],
    }


def run_feasibility(lat: float, lon: float, system: PVSystem,
                    weather: pd.DataFrame | None = None, start_year: int = HISTORY_START_YEAR,
                    end_year: int = HISTORY_END_YEAR) -> FeasibilityResult:
    weather = get_history(lat, lon, start_year, end_year) if weather is None else weather
    years = complete_years(weather)
    if not years:
        raise ValueError("no complete calendar year of weather data")
    weather = weather[weather.index.year.isin(years)]

    cal = load_calibration("hindcast")
    sim = simulate(weather, system, lat, lon)
    ac = (sim["ac"] * cal.k).clip(upper=system.ac_capacity_kw)

    annual = ac.groupby(ac.index.year).sum()
    p50 = float(annual.mean())
    iav = float(annual.std(ddof=1) / p50) if len(annual) > 1 and p50 > 0 else 0.0
    s_model, s_source = cal.k_std, SOURCE_UNCERTAINTY_PCT / 100
    s_lt = float(np.sqrt(s_model ** 2 + s_source ** 2 + iav ** 2 / 10))
    s_1y = float(np.sqrt(s_model ** 2 + s_source ** 2 + iav ** 2))
    p90 = p50 * (1 - Z90 * s_lt)
    p90_1y = p50 * (1 - Z90 * s_1y)

    by_ym = ac.groupby([ac.index.year, ac.index.month]).sum().unstack()
    monthly = pd.DataFrame({
        "month": MONTHS,
        "p50_kwh": by_ym.mean().to_numpy(),
        "low_year_kwh": by_ym.quantile(0.1).to_numpy(),
        "high_year_kwh": by_ym.quantile(0.9).to_numpy(),
        "poa_kwh_m2": sim["poa_global"].groupby(sim.index.month).sum().to_numpy() / 1000 / len(years),
        "ghi_kwh_m2": weather["ghi"].groupby(weather.index.month).sum().to_numpy() / 1000 / len(years),
        "mean_temp_c": weather["temp_air"].groupby(weather.index.month).mean().to_numpy(),
    })
    monthly["specific_yield"] = monthly["p50_kwh"] / system.capacity_kwp
    monthly["performance_ratio"] = monthly["specific_yield"] / monthly["poa_kwh_m2"]

    wf = loss_waterfall(sim, weather, system)
    final = wf["kwh_per_kwp"].iloc[-1]
    wf = pd.concat([wf, pd.DataFrame([{
        "step": f"Real-plant calibration (k = {cal.k:.3f})",
        "kwh_per_kwp": final * cal.k,
        "change_pct": round((cal.k - 1) * 100, 2),
    }])], ignore_index=True)

    res = FeasibilityResult(
        latitude=lat, longitude=lon, system=system,
        weather_source=weather.attrs.get("source", "unknown"), years=years,
        annual_by_year=annual, p50_kwh=p50, p90_kwh=p90, p90_1y_kwh=p90_1y,
        uncertainty={"model_pct": s_model * 100, "source_pct": s_source * 100,
                     "interannual_pct": iav * 100, "total_long_term_pct": s_lt * 100,
                     "total_single_year_pct": s_1y * 100, "calibration_k": cal.k},
        monthly=monthly, waterfall=wf, climate=_climate_summary(weather, sim, monthly.set_index(np.arange(1, 13))["p50_kwh"]),
        tilt=sim.attrs["tilt"], azimuth=sim.attrs["azimuth"],
        calibrated=cal.available,
    )
    res.verdict = make_verdict(res.specific_yield_p50, res.specific_yield_p90)
    return res


def best_orientation(lat: float, lon: float, system: PVSystem, weather: pd.DataFrame | None = None,
                     n_years: int = 3) -> pd.DataFrame:
    """Tilt sweep on the most recent `n_years` complete years (fast enough for the UI)."""
    weather = get_history(lat, lon, HISTORY_START_YEAR, HISTORY_END_YEAR) if weather is None else weather
    yrs = complete_years(weather)[-n_years:]
    w = weather[weather.index.year.isin(yrs)]
    w.attrs = weather.attrs
    return optimise_orientation(w, system, lat, lon)
