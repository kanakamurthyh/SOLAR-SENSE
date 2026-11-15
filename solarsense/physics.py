"""
Physics core: weather (WeatherFrame) -> PV power, using pvlib.

Chain:
    solar position -> plane-of-array transposition (Perez) -> incidence-angle
    (IAM) losses -> cell temperature (Faiman) -> DC power (PVWatts, temperature
    coefficient) -> system losses (soiling, wiring, ...) -> inverter efficiency
    and AC clipping.

Every stage is kept so the app can show a loss waterfall.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pvlib

from solarsense.config import (
    DEFAULT_ALBEDO,
    DEFAULT_CAPACITY_KWP,
    DEFAULT_DC_AC_RATIO,
    DEFAULT_GAMMA_PDC,
    DEFAULT_INVERTER_EFFICIENCY,
    DEFAULT_LOSSES_PCT,
)
from solarsense.weather import validate_weather


@dataclass(frozen=True)
class PVSystem:
    """A PV plant. `tilt=None` means 'use the latitude-based default'."""
    capacity_kwp: float = DEFAULT_CAPACITY_KWP
    dc_ac_ratio: float = DEFAULT_DC_AC_RATIO
    tilt: float | None = None
    azimuth: float | None = None          # degrees from north; None -> equator-facing
    tracking: str = "fixed"               # "fixed" | "single_axis"
    gamma_pdc: float = DEFAULT_GAMMA_PDC
    inverter_efficiency: float = DEFAULT_INVERTER_EFFICIENCY
    albedo: float = DEFAULT_ALBEDO
    losses_pct: dict = field(default_factory=lambda: dict(DEFAULT_LOSSES_PCT))

    def __post_init__(self):
        if self.capacity_kwp <= 0:
            raise ValueError("capacity_kwp must be positive")
        if not 0.8 <= self.dc_ac_ratio <= 2.0:
            raise ValueError("dc_ac_ratio must be within 0.8..2.0")
        if self.tracking not in ("fixed", "single_axis"):
            raise ValueError("tracking must be 'fixed' or 'single_axis'")

    @property
    def ac_capacity_kw(self) -> float:
        return self.capacity_kwp / self.dc_ac_ratio

    @property
    def system_loss_fraction(self) -> float:
        return float(pvlib.pvsystem.pvwatts_losses(**self.losses_pct)) / 100.0

    def resolved_tilt(self, lat: float) -> float:
        # Fixed-tilt annual optimum in India is close to latitude minus a few
        # degrees (monsoon clouds reduce summer weight); use 0.9*|lat|.
        return float(self.tilt) if self.tilt is not None else round(min(abs(lat) * 0.9, 40.0), 1)

    def resolved_azimuth(self, lat: float) -> float:
        if self.azimuth is not None:
            return float(self.azimuth)
        return 180.0 if lat >= 0 else 0.0


def simulate(weather: pd.DataFrame, system: PVSystem, lat: float | None = None,
             lon: float | None = None, losses: bool = True, clip: bool = True) -> pd.DataFrame:
    """
    Simulate hourly PV output. Returns a DataFrame aligned to `weather.index` with:
        solar_zenith, solar_elevation, aoi, clearsky_ghi, clearsky_index,
        poa_global, poa_effective, temp_cell,
        dc_stc (kW at 25 °C, before temperature), dc_gross (after temperature),
        dc_net (after system losses), ac_unclipped, ac (kW), clipped (kW)
    """
    validate_weather(weather)
    lat = weather.attrs.get("latitude") if lat is None else lat
    lon = weather.attrs.get("longitude") if lon is None else lon
    if lat is None or lon is None:
        raise ValueError("latitude/longitude required (pass them or set weather.attrs)")
    altitude = weather.attrs.get("elevation")
    altitude = 0.0 if altitude is None or not np.isfinite(altitude) else float(altitude)

    loc = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    times = weather.index
    sp = loc.get_solarposition(times, temperature=weather["temp_air"])
    zenith, azimuth = sp["apparent_zenith"], sp["azimuth"]

    if system.tracking == "single_axis":
        tr = pvlib.tracking.singleaxis(zenith, azimuth, axis_tilt=0, axis_azimuth=180,
                                       max_angle=60, backtrack=True, gcr=0.35)
        surface_tilt = tr["surface_tilt"].fillna(0.0)
        surface_azimuth = tr["surface_azimuth"].fillna(180.0)
    else:
        surface_tilt = system.resolved_tilt(lat)
        surface_azimuth = system.resolved_azimuth(lat)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt, surface_azimuth, zenith, azimuth,
        dni=weather["dni"], ghi=weather["ghi"], dhi=weather["dhi"],
        dni_extra=dni_extra, airmass=airmass, albedo=system.albedo, model="perez",
    ).fillna(0.0)
    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth, zenith, azimuth)

    iam_beam = pvlib.iam.physical(aoi)
    iam_diff = pvlib.iam.marion_diffuse("physical", np.asarray(surface_tilt, dtype=float))
    poa_eff = (poa["poa_direct"] * iam_beam
               + poa["poa_sky_diffuse"] * iam_diff["sky"]
               + poa["poa_ground_diffuse"] * iam_diff["ground"]).clip(lower=0).fillna(0.0)

    temp_cell = pvlib.temperature.faiman(poa["poa_global"], weather["temp_air"], weather["wind_speed"])

    pdc0_w = system.capacity_kwp * 1000.0
    dc_stc = poa_eff / 1000.0 * pdc0_w
    dc_gross = pvlib.pvsystem.pvwatts_dc(poa_eff, temp_cell, pdc0_w, system.gamma_pdc).clip(lower=0)
    loss_frac = system.system_loss_fraction if losses else 0.0
    dc_net = dc_gross * (1.0 - loss_frac)

    pac0_w = system.ac_capacity_kw * 1000.0
    ac_unclipped = dc_net * system.inverter_efficiency
    ac = ac_unclipped.clip(upper=pac0_w) if clip else ac_unclipped

    night = zenith >= 90
    out = pd.DataFrame({
        "solar_zenith": sp["zenith"],
        "solar_elevation": sp["elevation"],
        "aoi": aoi,
        "poa_global": poa["poa_global"],
        "poa_effective": poa_eff,
        "temp_cell": temp_cell,
        "dc_stc": dc_stc / 1000.0,
        "dc_gross": dc_gross / 1000.0,
        "dc_net": dc_net / 1000.0,
        "ac_unclipped": ac_unclipped / 1000.0,
        "ac": ac / 1000.0,
    }, index=times)
    energy_cols = ["poa_global", "poa_effective", "dc_stc", "dc_gross", "dc_net", "ac_unclipped", "ac"]
    out.loc[night, energy_cols] = 0.0
    out["clipped"] = out["ac_unclipped"] - out["ac"]

    cs = loc.get_clearsky(times, model="ineichen")
    out["clearsky_ghi"] = cs["ghi"]
    out["clearsky_index"] = np.where(cs["ghi"] > 20, weather["ghi"] / cs["ghi"].where(cs["ghi"] > 20), np.nan)
    out["clearsky_index"] = out["clearsky_index"].clip(0, 1.5)
    out.attrs.update(tilt=float(np.mean(surface_tilt)), azimuth=float(np.mean(surface_azimuth)),
                     latitude=lat, longitude=lon, system_loss_fraction=loss_frac)
    return out


def loss_waterfall(sim: pd.DataFrame, weather: pd.DataFrame, system: PVSystem) -> pd.DataFrame:
    """
    Annualised energy flow per kWp, from irradiance on the horizontal to AC
    delivered. Each row is the energy remaining after that step.
    """
    years = max(len(sim) / 8766.0, 1e-9)
    kwp = system.capacity_kwp
    ghi_kwh = weather["ghi"].sum() / 1000.0 / years
    poa_kwh = sim["poa_global"].sum() / 1000.0 / years
    eff_kwh = sim["poa_effective"].sum() / 1000.0 / years
    steps = [
        ("Horizontal irradiation (GHI)", ghi_kwh),
        ("Plane-of-array (tilt gain)", poa_kwh),
        ("After reflection (IAM)", eff_kwh),
        ("After temperature", sim["dc_gross"].sum() / kwp / years),
        ("After system losses", sim["dc_net"].sum() / kwp / years),
        ("After inverter", sim["ac_unclipped"].sum() / kwp / years),
        ("After clipping (AC delivered)", sim["ac"].sum() / kwp / years),
    ]
    df = pd.DataFrame(steps, columns=["step", "kwh_per_kwp"])
    df["change_pct"] = df["kwh_per_kwp"].pct_change().mul(100).round(2)
    return df


def optimise_orientation(weather: pd.DataFrame, system: PVSystem, lat: float, lon: float,
                         tilts=range(0, 46, 3), azimuths=None) -> pd.DataFrame:
    """Grid-search fixed tilt (and optionally azimuth) for maximum annual AC yield per kWp."""
    if azimuths is None:
        azimuths = [system.resolved_azimuth(lat)]
    rows = []
    for az in azimuths:
        for t in tilts:
            s = PVSystem(**{**system.__dict__, "tilt": float(t), "azimuth": float(az), "tracking": "fixed"})
            sim = simulate(weather, s, lat, lon)
            years = len(sim) / 8766.0
            rows.append({"tilt": t, "azimuth": az, "kwh_per_kwp": sim["ac"].sum() / s.capacity_kwp / years})
    return pd.DataFrame(rows).sort_values("kwh_per_kwp", ascending=False).reset_index(drop=True)
