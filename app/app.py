"""
SolarSense v2 - Streamlit app.

Run from project root:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from solarsense import __version__  # noqa: E402
from solarsense.calibration import load_calibration  # noqa: E402
from solarsense.config import (  # noqa: E402
    ATLAS_PATH,
    DEFAULT_CAPACITY_KWP,
    DEFAULT_DC_AC_RATIO,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_LOSSES_PCT,
    HISTORY_END_YEAR,
    HISTORY_START_YEAR,
    INDIA_LOCATIONS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from solarsense.feasibility import best_orientation, run_feasibility  # noqa: E402
from solarsense.forecast import run_forecast  # noqa: E402
from solarsense.physics import PVSystem  # noqa: E402
from solarsense.weather import WeatherError, fetch_openmeteo_forecast, get_history  # noqa: E402

st.set_page_config(page_title="SolarSense", page_icon="☀️", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>", unsafe_allow_html=True)

SUN, GOLD, GREEN, RED, BLUE, MUTED = "#F5A623", "#FFD166", "#06D6A0", "#EF476F", "#4EA8DE", "#8A8AA0"
PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#14141C",
                   font=dict(family="JetBrains Mono, monospace", size=12, color="#E8E8F0"),
                   margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_family="JetBrains Mono"))


def styled(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, height=height, **kw)
    fig.update_xaxes(gridcolor="#24242F")
    fig.update_yaxes(gridcolor="#24242F")
    return fig


def energy(kwh: float) -> str:
    return f"{kwh / 1e6:,.2f} GWh" if kwh >= 1e6 else (f"{kwh / 1e3:,.1f} MWh" if kwh >= 1e3 else f"{kwh:,.0f} kWh")


# ─────────────────────────────────────────────────────────────
# CACHED DATA ACCESS
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=32)
def cached_history(lat: float, lon: float) -> pd.DataFrame:
    return get_history(lat, lon, HISTORY_START_YEAR, HISTORY_END_YEAR)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=32)
def cached_forecast_weather(lat: float, lon: float) -> pd.DataFrame:
    return fetch_openmeteo_forecast(lat, lon)


@st.cache_data(show_spinner=False)
def load_locations() -> pd.DataFrame:
    return pd.read_csv(INDIA_LOCATIONS_PATH)


@st.cache_data(show_spinner=False)
def load_atlas() -> pd.DataFrame | None:
    return pd.read_csv(ATLAS_PATH) if ATLAS_PATH.exists() else None


def read_report(name: str):
    p = REPORTS_DIR / name
    if not p.exists():
        return None
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero">
  <span class="t1">SOLAR</span><span class="t2">SENSE</span><span class="tag">v{__version__}</span>
  <p>PHYSICS-FIRST · CALIBRATED ON REAL PLANTS · P50/P90 FEASIBILITY · PROBABILISTIC FORECAST</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────

locs = load_locations()
if "lat" not in st.session_state:
    st.session_state.lat, st.session_state.lon = DEFAULT_LATITUDE, DEFAULT_LONGITUDE

with st.sidebar:
    st.markdown("### 📍 Location")
    preset = st.selectbox("Indian city preset", ["Custom coordinates"] + locs["city"].tolist(), index=0,
                          help="Pick a city, or keep 'Custom' and type coordinates.")
    if preset != "Custom coordinates":
        row = locs[locs["city"] == preset].iloc[0]
        st.session_state.lat, st.session_state.lon = float(row.latitude), float(row.longitude)
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -60.0, 60.0, float(st.session_state.lat), 0.01, format="%.4f")
    lon = c2.number_input("Longitude", -180.0, 180.0, float(st.session_state.lon), 0.01, format="%.4f")
    st.session_state.lat, st.session_state.lon = lat, lon

    st.markdown("### ⚡ Plant")
    capacity = st.number_input("DC capacity (kWp)", 1.0, 2_000_000.0, DEFAULT_CAPACITY_KWP, 100.0,
                               help="Module nameplate. 1 MWp = 1000 kWp.")
    dc_ac = st.slider("DC/AC ratio", 1.0, 1.6, DEFAULT_DC_AC_RATIO, 0.05,
                      help="Indian utility plants typically use 1.2-1.4. Higher = more clipping, cheaper inverters.")
    tracking = st.radio("Mounting", ["Fixed tilt", "Single-axis tracker"], horizontal=True)
    tilt = azimuth = None
    if tracking == "Fixed tilt":
        auto = st.toggle("Auto tilt (0.9 × latitude)", value=True)
        if not auto:
            tilt = st.slider("Tilt (°)", 0, 60, int(round(abs(lat) * 0.9)))
            azimuth = st.slider("Azimuth (° from north, 180 = south)", 90, 270, 180 if lat >= 0 else 0)

    with st.expander("Loss assumptions (%)"):
        losses = {}
        for key, label in [("soiling", "Soiling"), ("shading", "Shading"), ("mismatch", "Mismatch"),
                           ("wiring", "DC wiring"), ("connections", "Connections"), ("lid", "LID"),
                           ("nameplate_rating", "Nameplate tolerance"), ("availability", "Availability")]:
            losses[key] = st.number_input(label, 0.0, 30.0, float(DEFAULT_LOSSES_PCT[key]), 0.5, key=f"loss_{key}")
        losses["snow"], losses["age"] = 0.0, 0.0

    run = st.button("☀  RUN ANALYSIS", type="primary", width="stretch")

system = PVSystem(capacity_kwp=capacity, dc_ac_ratio=dc_ac, tilt=tilt, azimuth=azimuth,
                  tracking="single_axis" if tracking == "Single-axis tracker" else "fixed", losses_pct=losses)
inputs_key = json.dumps({"lat": round(lat, 4), "lon": round(lon, 4), "system": asdict(system)}, sort_keys=True, default=str)

if run:
    st.session_state.pop("orientation", None)
    with st.spinner("🛰  Fetching 10 years of hourly weather and simulating…"):
        try:
            w = cached_history(round(lat, 4), round(lon, 4))
            st.session_state.feas = run_feasibility(lat, lon, system, weather=w)
            st.session_state.feas_error = None
        except (WeatherError, ValueError) as e:
            st.session_state.feas, st.session_state.feas_error = None, str(e)
    with st.spinner("🌤  Fetching 7-day forecast…"):
        try:
            st.session_state.fc = run_forecast(lat, lon, system, weather=cached_forecast_weather(round(lat, 4), round(lon, 4)))
            st.session_state.fc_error = None
        except (WeatherError, ValueError) as e:
            st.session_state.fc, st.session_state.fc_error = None, str(e)
    st.session_state.result_key = inputs_key

has_results = "feas" in st.session_state or "fc" in st.session_state
stale = has_results and st.session_state.get("result_key") != inputs_key

tabs = st.tabs(["☀ Feasibility", "🌤 7-Day Forecast", "🗺 India Atlas", "🔬 Model & Validation", "🤖 AI Analyst"])


def idle():
    st.info("Configure the plant in the sidebar and press **RUN ANALYSIS**. "
            "Or open the **India Atlas** tab to compare 36 cities instantly.")


def stale_note():
    if stale:
        st.warning("Inputs changed since the last run - press **RUN ANALYSIS** to refresh.", icon="⚠️")


# ─────────────────────────────────────────────────────────────
# TAB 1 - FEASIBILITY
# ─────────────────────────────────────────────────────────────

with tabs[0]:
    res = st.session_state.get("feas")
    if st.session_state.get("feas_error"):
        st.error(f"Feasibility failed: {st.session_state.feas_error}")
    elif res is None:
        idle()
    else:
        stale_note()
        v = res.verdict
        cls = "NOT" if v["label"] == "NOT VIABLE" else v["label"]
        reasons = "".join(f"<li>{r}</li>" for r in v["reasons"])
        st.markdown(f"""
        <div class="card">
          <div class="small">SITE {res.latitude:.4f}, {res.longitude:.4f} · {res.system.capacity_kwp:,.0f} kWp ·
          DC/AC {res.system.dc_ac_ratio:.2f} · tilt {res.tilt:.1f}° · {res.weather_source} {res.years[0]}–{res.years[-1]}</div>
          <div style="margin:.5rem 0"><span class="verdict v-{cls}">{v['label']}</span></div>
          <ul class="small" style="margin:0">{reasons}</ul>
        </div>""", unsafe_allow_html=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("P50 annual energy", energy(res.p50_kwh))
        k2.metric("P90 annual energy", energy(res.p90_kwh), f"-{(1 - res.p90_kwh / res.p50_kwh) * 100:.1f}% vs P50",
                  delta_color="off")
        k3.metric("Specific yield", f"{res.specific_yield_p50:,.0f}", "kWh/kWp/yr", delta_color="off")
        k4.metric("CUF (AC)", f"{res.cuf_ac_pct:.1f}%", f"CF (DC) {res.cf_dc_pct:.1f}%", delta_color="off")
        k5.metric("Perf. ratio", f"{res.monthly['performance_ratio'].mean():.2f}")
        left, right = st.columns([3, 2])
        with left:
            m = res.monthly
            fig = go.Figure()
            fig.add_bar(x=m["month"], y=m["p50_kwh"] / 1e3, name="P50 (10-yr mean)", marker_color=SUN,
                        error_y=dict(type="data", symmetric=False, array=(m["high_year_kwh"] - m["p50_kwh"]) / 1e3,
                                     arrayminus=(m["p50_kwh"] - m["low_year_kwh"]) / 1e3, color=MUTED))
            fig.add_scatter(x=m["month"], y=m["performance_ratio"], name="Performance ratio", yaxis="y2",
                            mode="lines+markers", line=dict(color=GREEN))
            styled(fig, title="Monthly energy (MWh) · whiskers = 10th–90th percentile year",
                   yaxis=dict(title="MWh"), yaxis2=dict(title="PR", overlaying="y", side="right", range=[0.5, 1],
                                                        showgrid=False), legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, width="stretch")
        with right:
            wf = res.waterfall
            base = wf["kwh_per_kwp"].to_numpy()
            measures = ["absolute"] + ["relative"] * (len(wf) - 1)
            vals = [base[0]] + list(np.diff(base))
            fig = go.Figure(go.Waterfall(x=wf["step"], y=vals, measure=measures,
                                         increasing=dict(marker_color=GREEN), decreasing=dict(marker_color=RED),
                                         totals=dict(marker_color=SUN), connector=dict(line=dict(color=MUTED)),
                                         text=[f"{v:+.0f}" if i else f"{v:.0f}" for i, v in enumerate(vals)],
                                         textposition="outside"))
            fig.add_bar(x=[wf["step"].iloc[-1]], y=[base[-1]], marker_color=SUN, showlegend=False, opacity=0.35)
            styled(fig, height=380, title="Energy flow (kWh per kWp per year)", showlegend=False)
            fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
            st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        with a:
            yrs = res.annual_by_year
            fig = go.Figure(go.Bar(x=yrs.index.astype(str), y=yrs / res.system.capacity_kwp, marker_color=BLUE))
            fig.add_hline(y=res.specific_yield_p50, line_dash="dash", line_color=SUN, annotation_text="P50")
            fig.add_hline(y=res.specific_yield_p90, line_dash="dot", line_color=RED, annotation_text="P90")
            styled(fig, height=320, title="Year-by-year specific yield (kWh/kWp)")
            fig.update_yaxes(range=[res.specific_yield_p90 * 0.85, yrs.max() / res.system.capacity_kwp * 1.05])
            st.plotly_chart(fig, width="stretch")
        with b:
            u = res.uncertainty
            parts = pd.DataFrame({"component": ["Plant-to-plant (real plants)", "Weather source", "Inter-annual",
                                                "Total (long-term)"],
                                  "sigma_pct": [u["model_pct"], u["source_pct"], u["interannual_pct"] / np.sqrt(10),
                                                u["total_long_term_pct"]]})
            fig = px.bar(parts, x="sigma_pct", y="component", orientation="h", text=parts["sigma_pct"].round(1),
                         color="component", color_discrete_sequence=[BLUE, BLUE, BLUE, SUN])
            styled(fig, height=320, title="Uncertainty budget (1σ, % of energy)", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        c = res.climate
        st.markdown(f"""<div class="card small">
        <b>Climate</b> · GHI {c['ghi_kwh_m2_yr']:,.0f} kWh/m²/yr · clear-sky index {c['mean_clearsky_index']:.2f}
        (monsoon {c['monsoon_clearsky_index']:.2f}) · monsoon output dip {c['monsoon_dip_pct']:+.0f}% ·
        daytime air {c['mean_daytime_temp_c']:.1f} °C, cells {c['mean_cell_temp_c']:.1f} °C ·
        best {c['best_month']}, worst {c['worst_month']}
        {'' if res.calibrated else '<br>⚠️ Real-plant calibration not loaded - run training to enable it.'}
        </div>""", unsafe_allow_html=True)

        if res.system.tracking == "fixed":
            if st.button("📐 Optimise tilt for this site"):
                with st.spinner("Sweeping tilt angles…"):
                    w = cached_history(round(res.latitude, 4), round(res.longitude, 4))
                    st.session_state.orientation = best_orientation(res.latitude, res.longitude, res.system, weather=w)
            o = st.session_state.get("orientation")
            if o is not None:
                o = o.sort_values("tilt")
                best = o.loc[o["kwh_per_kwp"].idxmax()]
                fig = go.Figure(go.Scatter(x=o["tilt"], y=o["kwh_per_kwp"], mode="lines+markers", line=dict(color=SUN)))
                fig.add_vline(x=best["tilt"], line_dash="dash", line_color=GREEN, annotation_text=f"best {best['tilt']:.0f}°")
                fig.add_vline(x=res.tilt, line_dash="dot", line_color=MUTED, annotation_text="current")
                styled(fig, height=300, title="Physics yield vs tilt (kWh/kWp/yr, before calibration)", xaxis_title="Tilt (°)")
                st.plotly_chart(fig, width="stretch")

        report = {"inputs": json.loads(inputs_key), "p50_kwh": res.p50_kwh, "p90_kwh": res.p90_kwh,
                  "p90_1y_kwh": res.p90_1y_kwh, "specific_yield_p50": res.specific_yield_p50,
                  "cuf_ac_pct": res.cuf_ac_pct, "uncertainty": res.uncertainty, "climate": res.climate,
                  "verdict": res.verdict, "monthly": res.monthly.to_dict(orient="records"),
                  "waterfall": res.waterfall.to_dict(orient="records")}
        st.download_button("⬇ Download feasibility report (JSON)", json.dumps(report, indent=2, default=float),
                           file_name=f"solarsense_{res.latitude:.3f}_{res.longitude:.3f}.json", mime="application/json")

# ─────────────────────────────────────────────────────────────
# TAB 2 - FORECAST
# ─────────────────────────────────────────────────────────────

with tabs[1]:
    fc = st.session_state.get("fc")
    if st.session_state.get("fc_error"):
        st.error(f"Forecast failed: {st.session_state.fc_error}")
    elif fc is None:
        idle()
    else:
        stale_note()
        tot = fc.total
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Forecast P50 (full days)", energy(tot["p50"]))
        k2.metric("P10 – P90", f"{energy(tot['p10'])} – {energy(tot['p90'])}")
        k3.metric("Best day", str(fc.daily.loc[fc.daily['p50_kwh'].idxmax(), 'date']) if len(fc.daily) else "—")
        k4.metric("Peak power (P50)", f"{fc.hourly['p50_kw'].max():,.0f} kW", f"AC limit {fc.ac_capacity_kw:,.0f} kW",
                  delta_color="off")
        for n in fc.notes:
            st.warning(n)

        h = fc.hourly
        x = h["local_time"].dt.tz_localize(None)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
        fig.add_scatter(x=x, y=h["p90_kw"], line=dict(width=0), showlegend=False, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=x, y=h["p10_kw"], fill="tonexty", fillcolor="rgba(245,166,35,0.22)", line=dict(width=0),
                        name="P10–P90 (80% band)", row=1, col=1)
        fig.add_scatter(x=x, y=h["p50_kw"], line=dict(color=SUN, width=2.5), name="P50", row=1, col=1)
        fig.add_scatter(x=x, y=h["physics_kw"], line=dict(color=MUTED, width=1, dash="dot"), name="Physics (uncalibrated)",
                        row=1, col=1)
        fig.add_bar(x=x, y=h["cloud_cover"], marker_color=BLUE, name="Cloud cover %", opacity=0.7, row=2, col=1)
        styled(fig, height=520, title="Hourly power forecast (local time)", legend=dict(orientation="h", y=1.08))
        fig.update_yaxes(title_text="kW", row=1, col=1)
        fig.update_yaxes(title_text="cloud %", range=[0, 100], row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        d = fc.daily
        fig = go.Figure(go.Bar(x=d["date"].astype(str), y=d["p50_kwh"] / 1e3, marker_color=SUN, name="P50",
                               error_y=dict(type="data", symmetric=False, array=(d["p90_kwh"] - d["p50_kwh"]) / 1e3,
                                            arrayminus=(d["p50_kwh"] - d["p10_kwh"]) / 1e3, color=MUTED)))
        styled(fig, height=320, title="Daily energy (MWh) · whiskers = P10–P90")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(d.rename(columns={"date": "Date", "p10_kwh": "P10 kWh", "p50_kwh": "P50 kWh", "p90_kwh": "P90 kWh",
                                       "ghi_kwh_m2": "GHI kWh/m²", "cloud_cover_mean": "Cloud %", "peak_kw": "Peak kW",
                                       "clearsky_index": "Clear-sky idx"}).round(2),
                     hide_index=True, width="stretch")
        st.caption("Bands are conformally calibrated on real US plants the model never saw (80 % hourly and daily "
                   "coverage) and widen with lead time. For clear dry-season Indian days they are likely conservative.")

# ─────────────────────────────────────────────────────────────
# TAB 3 - ATLAS
# ─────────────────────────────────────────────────────────────

with tabs[2]:
    atlas = load_atlas()
    if atlas is None:
        st.info("Atlas not built yet - run `python3 scripts/build_india_atlas.py`.")
    else:
        st.markdown("Calibrated P50 specific yield for a 1 kWp fixed-tilt plant (DC/AC 1.25, default Indian losses), "
                    "ERA5 2020–2022. **Click a city on the map** to load it into the sidebar.")
        fig = px.scatter_map(atlas, lat="latitude", lon="longitude", color="specific_yield_p50", size="specific_yield_p50",
                             size_max=22, hover_name="city",
                             hover_data={"state": True, "specific_yield_p50": True, "cuf_ac_pct": True,
                                         "monsoon_dip_pct": True, "latitude": False, "longitude": False},
                             color_continuous_scale="YlOrRd", zoom=3.4, center=dict(lat=22.5, lon=81),
                             map_style="carto-darkmatter")
        styled(fig, height=560, coloraxis_colorbar=dict(title="kWh/kWp"))
        sel = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="atlas_map")
        pts = (sel or {}).get("selection", {}).get("points", [])
        if pts:
            city = pts[0].get("hovertext")
            r = atlas[atlas["city"] == city]
            if len(r):
                st.session_state.lat, st.session_state.lon = float(r.latitude.iloc[0]), float(r.longitude.iloc[0])
                st.success(f"Loaded **{city}** ({st.session_state.lat:.3f}, {st.session_state.lon:.3f}). "
                           "Set the sidebar preset to 'Custom coordinates' if needed, then press RUN ANALYSIS.")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(atlas[["rank", "city", "state", "region", "specific_yield_p50", "cuf_ac_pct", "best_month",
                                "worst_month", "monsoon_dip_pct", "optimal_tilt"]],
                         hide_index=True, width="stretch", height=420)
        with c2:
            reg = atlas.groupby("region")["specific_yield_p50"].mean().sort_values()
            fig = go.Figure(go.Bar(x=reg.values, y=reg.index, orientation="h", marker_color=SUN))
            styled(fig, height=420, title="Mean yield by region (kWh/kWp)")
            fig.update_xaxes(range=[reg.min() * 0.9, reg.max() * 1.02])
            st.plotly_chart(fig, width="stretch")
        pick = st.multiselect("Compare monthly profiles", atlas["city"].tolist(), default=atlas["city"].head(3).tolist())
        if pick and st.button("Compare selected cities"):
            from scripts.validate_physics_vs_pvgis import load_om
            cal = load_calibration("hindcast")
            fig = go.Figure()
            for city in pick:
                r = atlas[atlas["city"] == city].iloc[0]
                w = load_om(r.location_id, r.latitude, r.longitude)
                from solarsense.physics import simulate
                s = simulate(w, PVSystem(capacity_kwp=1.0), r.latitude, r.longitude)["ac"] * cal.k
                mon = s.groupby(s.index.month).sum() / 3
                fig.add_scatter(x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                                y=mon.values, name=city, mode="lines+markers")
            styled(fig, height=360, title="Monthly specific yield (kWh/kWp)")
            st.plotly_chart(fig, width="stretch")

# ─────────────────────────────────────────────────────────────
# TAB 4 - MODEL & VALIDATION
# ─────────────────────────────────────────────────────────────

with tabs[3]:
    hc, fcal = load_calibration("hindcast"), load_calibration("forecast")
    st.markdown("""
**How SolarSense estimates energy**
1. **Physics (pvlib):** solar position → Perez plane-of-array transposition → reflection (IAM) → Faiman cell temperature →
   PVWatts DC → loss chain → inverter + clipping.
2. **Real-plant calibration:** the physics model is compared with **5 real, metered US PV plants** (NREL PVDAQ, 2015–2024).
   The mean measured/physics ratio *k* corrects optimism; its plant-to-plant spread sets the P90.
3. **Uncertainty model (ML):** quantile XGBoost + Mondrian conformal calibration gives P10–P90 bands that cover 80 % of
   real outcomes on plants the model never saw.
""")
    if not hc.available:
        st.warning("Model cards not found - run `python3 -m training.build_training_set && python3 -m training.train`.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Calibration k", f"{hc.k:.3f}", f"±{hc.k_std * 100:.1f}% plant-to-plant", delta_color="off")
        c2.metric("Hourly band coverage", f"{hc.card['band_calibration']['hourly_coverage_pct']:.0f}%", "target 80%",
                  delta_color="off")
        if fcal.available:
            c3.metric("Forecast skill vs persistence", f"{fcal.card.get('physics_skill_vs_persistence_pct', 0):.0f}%",
                      "out-of-plant", delta_color="off")
            c4.metric("Forecast hourly nRMSE", f"{fcal.card['cv_summary']['physics']['mean_nrmse_pct']:.1f}%",
                      "of capacity", delta_color="off")
        kk = pd.Series(hc.card["plant_energy_factor_k"]).rename("k").reset_index().rename(columns={"index": "plant"})
        fig = px.bar(kk, x="plant", y="k", text=kk["k"].round(3), color_discrete_sequence=[BLUE])
        fig.add_hline(y=1, line_color=MUTED, line_dash="dot")
        fig.add_hline(y=hc.k, line_color=SUN, annotation_text=f"mean k = {hc.k:.3f}")
        styled(fig, height=320, title="Measured ÷ physics energy per real plant (age-corrected)")
        fig.update_yaxes(range=[0.8, 1.1])
        st.plotly_chart(fig, width="stretch")

    metrics = read_report("ml_cv_metrics.csv")
    if metrics is not None:
        st.markdown("**Leave-one-plant-out validation** (hourly, % of plant capacity)")
        piv = metrics.pivot_table(index=["source", "held_out_plant"], columns="method", values="nrmse_pct").round(2)
        st.dataframe(piv, width="stretch")

    lt = read_report("forecast_lead_time.csv")
    if lt is not None:
        fig = px.line(lt, x="lead_day", y="nrmse_pct", color="plant", markers=True)
        styled(fig, height=300, title="Forecast error vs lead time (physics on archived NWP)", xaxis_title="Lead day",
               yaxis_title="nRMSE %")
        st.plotly_chart(fig, width="stretch")

    pv = read_report("physics_vs_pvgis.csv")
    if pv is not None:
        fig = px.scatter(pv, x="yield_pvgis", y="yield_ours", color="region", hover_name="city")
        lo, hi = pv[["yield_pvgis", "yield_ours"]].min().min() * 0.97, pv[["yield_pvgis", "yield_ours"]].max().max() * 1.03
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False)
        styled(fig, height=380, title="Golden test: SolarSense physics vs PVGIS, 36 Indian cities (kWh/kWp/yr, no losses)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Mean difference {pv['yield_diff_pct'].mean():+.1f}%, MAE {pv['yield_diff_pct'].abs().mean():.1f}%. "
                   "Both use ERA5 irradiance processed independently; the spread sets the weather-source uncertainty.")

    preds = read_report("cv_predictions.parquet")
    if preds is not None:
        st.markdown("**Out-of-plant predictions vs real measurements**")
        c1, c2 = st.columns(2)
        plant = c1.selectbox("Plant", sorted(preds["plant"].unique()))
        source = c2.selectbox("Weather", ["forecast", "hindcast"])
        p = preds[(preds["plant"] == plant) & (preds["source"] == source)].sort_values("time")
        if len(p):
            days = sorted(p["time"].dt.floor("D").unique())
            start = st.select_slider("Week starting", options=days[::7], value=days[len(days) // 2 // 7 * 7])
            q = p[(p["time"] >= start) & (p["time"] < start + pd.Timedelta(days=7))].set_index("time")
            full = q.reindex(pd.date_range(start, start + pd.Timedelta(days=7), freq="h", tz="UTC") + pd.Timedelta(minutes=30))
            fig = go.Figure()
            fig.add_scatter(x=full.index, y=full["phys_lossy_pu"], name="Physics", line=dict(color=SUN))
            fig.add_scatter(x=full.index, y=full["target_norm"], name="Measured", line=dict(color=GREEN))
            styled(fig, height=320, title="Output per kWp (normalised), UTC")
            st.plotly_chart(fig, width="stretch")

    with st.expander("Model cards (raw)"):
        for name in ("hindcast", "forecast"):
            p = MODELS_DIR / f"{name}_model_card.json"
            if p.exists():
                card = json.loads(p.read_text())
                card.pop("cv_per_plant", None)
                st.json(card, expanded=False)

# ─────────────────────────────────────────────────────────────
# TAB 5 - AI ANALYST
# ─────────────────────────────────────────────────────────────

with tabs[4]:
    from solarsense import agent

    if not agent.is_available():
        st.markdown(agent.setup_instructions())
    else:
        st.markdown("Ask about any Indian site. The analyst **calls SolarSense's physics, forecast and atlas tools** "
                    "and only quotes numbers those tools return.")
        if "chat" not in st.session_state:
            st.session_state.chat = []
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        q = st.chat_input("e.g. Which yields more for a 5 MW plant, Bikaner or Coimbatore, and how big is the monsoon dip?")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.status("Analysing…", expanded=False) as status:
                    answer, trace = agent.ask(st.session_state.chat)
                    for t in trace:
                        status.write(f"🔧 `{t['tool']}` {json.dumps(t['input'])[:160]}")
                    status.update(label=f"Done · {len(trace)} tool calls", state="complete")
                st.markdown(answer)
            st.session_state.chat.append({"role": "assistant", "content": answer})

st.markdown(f"<div class='small' style='margin-top:2.5rem;border-top:1px solid #222;padding-top:.8rem'>"
            f"SolarSense v{__version__} · Kanakamurthy H, IIT Bombay · Weather: Open-Meteo (ERA5 © ECMWF), NASA POWER · "
            f"Real plants: NREL PVDAQ · Physics: pvlib</div>", unsafe_allow_html=True)
