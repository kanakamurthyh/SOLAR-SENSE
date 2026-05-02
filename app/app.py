"""
app.py — SolarSense Streamlit Application
Run: streamlit run app/app.py
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "utils"))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from model import load_model
from feasibility import run_forecast, run_feasibility
from evaluate import plot_feature_importance

st.set_page_config(
    page_title="SolarSense", page_icon="☀️",
    layout="wide", initial_sidebar_state="expanded"
)

# Sidebar toggle state
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# Toggle button — floats top-left of main area
col_toggle, col_rest = st.columns([0.05, 0.95])
with col_toggle:
    if st.button("◀" if st.session_state.sidebar_open else "▶",
                 key="sidebar_toggle",
                 help="Toggle sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

# Apply sidebar visibility based on state
if st.session_state.sidebar_open:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        display: block !important;
        visibility: visible !important;
        transform: translateX(0) !important;
        min-width: 370px !important;
        max-width: 420px !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
:root {
    --sun:#F5A623; --sun-dim:#C47D0E; --gold:#FFD166;
    --green:#06D6A0; --red:#EF476F;
    --bg:#0A0A0F; --bg2:#111118; --bg3:#1A1A24;
    --border:rgba(245,166,35,0.15); --text:#E8E8F0; --muted:#6B6B80;
}
html,body,[class*="css"]{font-family:'Syne',sans-serif!important;background:var(--bg)!important;color:var(--text)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 2rem 4rem!important;max-width:1400px!important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-thumb{background:var(--sun-dim);border-radius:2px;}
[data-testid="stSidebar"]{background:var(--bg2)!important;border-right:1px solid var(--border)!important;}
[data-testid="stSidebarCollapsedControl"],button[data-testid="stSidebarCollapsedControl"]{background:var(--sun)!important;border-radius:0 8px 8px 0!important;width:28px!important;min-height:64px!important;display:flex!important;align-items:center!important;justify-content:center!important;visibility:visible!important;opacity:1!important;}
[data-testid="stSidebarCollapsedControl"] svg{fill:#000!important;}
button[kind="header"]{color:var(--sun)!important;}
input,.stNumberInput input{background:var(--bg3)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:6px!important;font-family:'JetBrains Mono',monospace!important;}
input:focus{border-color:var(--sun)!important;box-shadow:0 0 0 2px rgba(245,166,35,0.2)!important;}
[data-testid="stSelectbox"] > div > div{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:6px!important;}
[data-testid="stToggle"] span[data-checked="true"]{background:var(--sun)!important;}
.stButton button{background:linear-gradient(135deg,var(--sun),var(--sun-dim))!important;color:#000!important;font-family:'Syne',sans-serif!important;font-weight:700!important;font-size:15px!important;border:none!important;border-radius:8px!important;padding:0.6rem 2rem!important;letter-spacing:0.05em!important;transition:all 0.2s!important;width:100%!important;}
.stButton button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 24px rgba(245,166,35,0.35)!important;}
[data-testid="stTabs"] [role="tab"]{font-family:'Syne',sans-serif!important;font-weight:600!important;color:var(--muted)!important;padding:0.5rem 1.5rem!important;}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--sun)!important;border-bottom:2px solid var(--sun)!important;}
[data-testid="metric-container"]{background:var(--bg3)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:1rem 1.2rem!important;}
[data-testid="metric-container"] label{color:var(--muted)!important;font-size:11px!important;letter-spacing:0.1em!important;text-transform:uppercase!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:var(--gold)!important;font-size:1.6rem!important;font-weight:700!important;}
hr{border-color:var(--border)!important;}
.sc{background:var(--bg3);border:1px solid var(--border);border-radius:16px;padding:1.5rem 2rem;margin:1rem 0;}
.sc-a{border-left:3px solid var(--sun);}
.vv{display:inline-block;background:rgba(6,214,160,0.12);color:var(--green);border:1px solid rgba(6,214,160,0.3);border-radius:99px;padding:0.4rem 1.5rem;font-weight:700;font-size:1rem;letter-spacing:0.1em;}
.vm{display:inline-block;background:rgba(245,166,35,0.12);color:var(--sun);border:1px solid rgba(245,166,35,0.3);border-radius:99px;padding:0.4rem 1.5rem;font-weight:700;font-size:1rem;letter-spacing:0.1em;}
.vn{display:inline-block;background:rgba(239,71,111,0.12);color:var(--red);border:1px solid rgba(239,71,111,0.3);border-radius:99px;padding:0.4rem 1.5rem;font-weight:700;font-size:1rem;letter-spacing:0.1em;}
.mono{font-family:'JetBrains Mono',monospace!important;font-size:12px;color:var(--muted);}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#111118","axes.facecolor":"#1A1A24","axes.edgecolor":"#2A2A38",
    "axes.labelcolor":"#E8E8F0","text.color":"#E8E8F0","xtick.color":"#6B6B80","ytick.color":"#6B6B80",
    "grid.color":"#2A2A38","grid.linewidth":0.5,"axes.grid":True,
    "axes.spines.top":False,"axes.spines.right":False,"font.family":"monospace","figure.dpi":130,
})

def sfig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def vhtml(v):
    if v=="VIABLE":   return '<span class="vv">✦ VIABLE</span>'
    if v=="MARGINAL": return '<span class="vm">◈ MARGINAL</span>'
    return '<span class="vn">✕ NOT VIABLE</span>'

def mrow(data):
    cols = st.columns(len(data))
    for col, item in zip(cols, data):
        with col:
            if len(item) == 4:
                l, v, d, dc = item
                st.metric(l, v, d, delta_color=dc)
            else:
                l, v, d = item
                st.metric(l, v, d)

st.markdown("""
<div style="padding:2.5rem 0 1rem;border-bottom:1px solid rgba(245,166,35,0.12);margin-bottom:0.5rem;">
  <div style="display:flex;align-items:baseline;gap:1rem;">
    <span style="font-size:2.8rem;font-weight:800;letter-spacing:-0.02em;color:#F5A623;">SOLAR</span>
    <span style="font-size:2.8rem;font-weight:800;letter-spacing:-0.02em;color:#E8E8F0;">SENSE</span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#6B6B80;border:1px solid rgba(245,166,35,0.2);padding:2px 10px;border-radius:99px;margin-left:8px;">v1.0</span>
  </div>
  <p style="color:#6B6B80;font-size:13px;letter-spacing:0.08em;margin:0.3rem 0 0;">
    SOLAR PLANT FEASIBILITY · POWER FORECASTING · INVESTMENT ANALYSIS
  </p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_model():
    try:    return *load_model(), None
    except FileNotFoundError as e: return None, None, str(e)

with st.spinner("⚡ Loading model..."):
    model, metadata, load_error = get_model()

if load_error:
    st.error(f"Model not found. Run `python src/model.py` first.\n\n{load_error}")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="padding:1.2rem 0 0.8rem;"><div style="font-size:10px;letter-spacing:0.14em;color:#6B6B80;">☀ SOLARSENSE CONFIGURATION</div></div>', unsafe_allow_html=True)

    # CHANGE 1: Removed quick-select dropdown — coordinates only
    st.markdown("**📍 Location**")
    c1,c2 = st.columns(2)
    with c1: lat = st.number_input("Latitude",  value=26.91, format="%.4f", min_value=-90.0,  max_value=90.0)
    with c2: lon = st.number_input("Longitude", value=75.78, format="%.4f", min_value=-180.0, max_value=180.0)

    st.markdown(f"""
    <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                color:#F5A623;background:#1A1A24;border-radius:6px;
                padding:6px 10px;margin:4px 0 8px;">
        {lat:.4f}°N  {lon:.4f}°E
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**⚡ Plant Specifications**")
    capacity_kw = st.number_input("Capacity (kW)", value=1000, min_value=1, max_value=500000, step=100)
    st.markdown(f'<div class="mono" style="margin-bottom:4px;">{capacity_kw/1000:.3f} MW  ·  {capacity_kw:,} kW</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("**💰 Financial Analysis**")
    st.markdown('<div style="font-size:11px;color:#6B6B80;margin-bottom:6px;">Optional — toggle off for energy-only analysis</div>', unsafe_allow_html=True)
    include_financials = st.toggle("Include financial analysis", value=True)

    cost_inr = None
    tariff   = None
    if include_financials:
        cost_lakhs = st.number_input("Installation cost (₹ Lakhs)", value=50.0, min_value=0.1, max_value=100000.0, step=5.0)
        tariff     = st.number_input("Electricity tariff (₹/kWh)", value=7.0,  min_value=1.0, max_value=25.0,    step=0.5)
        cost_inr   = cost_lakhs * 100_000
        st.markdown(f'<div class="mono">Total cost: ₹{cost_inr/10000000:.2f} Cr</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:12px;color:#6B6B80;padding:4px 0;">Monthly output, capacity factor, and seasonal profile will still be shown.</div>', unsafe_allow_html=True)

    st.divider()
    run_btn = st.button("☀  RUN ANALYSIS", use_container_width=True)

    m_info = metadata["metrics"]
    st.markdown(f"""
    <div style="margin-top:1.5rem;padding:1rem;background:#0A0A0F;border-radius:10px;border:1px solid rgba(245,166,35,0.12);">
        <div style="font-size:10px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:6px;">MODEL</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#F5A623;margin-bottom:4px;">{metadata['model_name']}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6B6B80;">R² = <span style="color:#06D6A0;">{m_info['r2']:.4f}</span></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6B6B80;">nRMSE = <span style="color:#06D6A0;">{m_info['nrmse_pct']:.2f}%</span></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6B6B80;">Skill = <span style="color:#06D6A0;">{m_info['skill_score']:.1f}%</span></div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#6B6B80;">Feat = <span style="color:#06D6A0;">{len(metadata['features'])}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────
t1,t2,t3 = st.tabs(["⬡  FEASIBILITY ANALYSIS","◈  7-DAY FORECAST","▲  MODEL DIAGNOSTICS"])

# CHANGE 2: Solar panel SVG
_style = (
    "<style>"
    "@keyframes pg{0%,100%{filter:drop-shadow(0 0 8px rgba(245,166,35,.3))}50%{filter:drop-shadow(0 0 22px rgba(245,166,35,.7))}}"
    "@keyframes rp{0%,100%{opacity:.4}50%{opacity:1}}"
    "@keyframes ii{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}"
    ".iw{display:flex;flex-direction:column;align-items:center;padding:2.5rem 2rem;gap:1.8rem;animation:ii .7s ease forwards;}"
    ".psv{animation:pg 3s ease-in-out infinite;}"
    ".sr{animation:rp 2.5s ease-in-out infinite;}"
    ".it h2{font-size:1.5rem;font-weight:800;color:#F5A623;letter-spacing:-.01em;margin:0 0 .3rem;text-align:center;}"
    ".it p{color:#6B6B80;font-size:13px;margin:0;text-align:center;}"
    ".is{display:flex;gap:1rem;flex-wrap:wrap;justify-content:center;}"
    ".is div{display:flex;align-items:center;gap:.5rem;background:#1A1A24;border:1px solid rgba(245,166,35,.15);border-radius:99px;padding:.35rem .9rem;font-size:11px;color:#9090A8;}"
    ".is div b{color:#F5A623;}"
    "</style>"
)
_svg = (
    '<svg class="psv" width="360" height="270" viewBox="0 0 360 270" xmlns="http://www.w3.org/2000/svg">'
    "<defs>"
    '<radialGradient id="sg" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#FFD166" stop-opacity="1"/><stop offset="70%" stop-color="#F5A623" stop-opacity="0.6"/><stop offset="100%" stop-color="#F5A623" stop-opacity="0"/></radialGradient>'
    '<linearGradient id="pf" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#1E3A5F"/><stop offset="100%" stop-color="#0D1F33"/></linearGradient>'
    '<linearGradient id="pt" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#2A4A70"/><stop offset="100%" stop-color="#1A3050"/></linearGradient>'
    '<linearGradient id="ps" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stop-color="#0D1F33"/><stop offset="100%" stop-color="#060F1A"/></linearGradient>'
    '<pattern id="cg" x="0" y="0" width="19" height="23" patternUnits="userSpaceOnUse"><rect width="19" height="23" fill="none"/><line x1="0" y1="0" x2="19" y2="0" stroke="rgba(100,160,220,0.22)" stroke-width="0.8"/><line x1="0" y1="0" x2="0" y2="23" stroke="rgba(100,160,220,0.22)" stroke-width="0.8"/></pattern>'
    '<linearGradient id="gl" x1="0%" y1="0%" x2="60%" y2="60%"><stop offset="0%" stop-color="white" stop-opacity="0.13"/><stop offset="100%" stop-color="white" stop-opacity="0"/></linearGradient>'
    '<radialGradient id="gsh" cx="50%" cy="30%" r="50%"><stop offset="0%" stop-color="#000" stop-opacity="0.35"/><stop offset="100%" stop-color="#000" stop-opacity="0"/></radialGradient>'
    '<clipPath id="pc"><polygon points="125,98 265,82 265,198 125,214"/></clipPath>'
    "</defs>"
    '<g class="sr">'
    '<circle cx="54" cy="48" r="24" fill="url(#sg)"/>'
    '<line x1="54" y1="16" x2="54" y2="7" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>'
    '<line x1="54" y1="80" x2="54" y2="89" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>'
    '<line x1="22" y1="48" x2="13" y2="48" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>'
    '<line x1="86" y1="48" x2="95" y2="48" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.7"/>'
    '<line x1="32" y1="26" x2="25" y2="19" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>'
    '<line x1="76" y1="26" x2="83" y2="19" stroke="#F5A623" stroke-width="1.5" stroke-linecap="round" opacity="0.5"/>'
    '<circle cx="54" cy="48" r="15" fill="#FFD166" opacity="0.95"/>'
    '<circle cx="54" cy="48" r="8" fill="#FFF5C0"/>'
    "</g>"
    '<line x1="75" y1="62" x2="132" y2="112" stroke="#F5A623" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.35"/>'
    '<line x1="71" y1="68" x2="163" y2="124" stroke="#F5A623" stroke-width="0.8" stroke-dasharray="4 3" opacity="0.25"/>'
    '<ellipse cx="205" cy="252" rx="112" ry="11" fill="url(#gsh)"/>'
    '<polygon points="125,98 265,82 290,103 150,119" fill="url(#pt)"/>'
    '<polygon points="265,82 290,103 290,218 265,197" fill="url(#ps)"/>'
    '<polygon points="125,98 265,82 265,197 125,214" fill="url(#pf)"/>'
    '<rect x="125" y="82" width="140" height="132" fill="url(#cg)" clip-path="url(#pc)"/>'
    '<line x1="125" y1="122" x2="265" y2="106" stroke="rgba(150,200,255,0.3)" stroke-width="1" clip-path="url(#pc)"/>'
    '<line x1="125" y1="145" x2="265" y2="129" stroke="rgba(150,200,255,0.3)" stroke-width="1" clip-path="url(#pc)"/>'
    '<line x1="125" y1="168" x2="265" y2="152" stroke="rgba(150,200,255,0.3)" stroke-width="1" clip-path="url(#pc)"/>'
    '<line x1="125" y1="191" x2="265" y2="175" stroke="rgba(150,200,255,0.3)" stroke-width="1" clip-path="url(#pc)"/>'
    '<polygon points="125,98 265,82 265,197 125,214" fill="url(#gl)"/>'
    '<polygon points="125,98 265,82 265,197 125,214" fill="none" stroke="#3A6090" stroke-width="1.5"/>'
    '<polygon points="125,98 265,82 290,103 150,119" fill="none" stroke="#3A6090" stroke-width="1.5"/>'
    '<polygon points="265,82 290,103 290,218 265,197" fill="none" stroke="#243A54" stroke-width="1.5"/>'
    '<line x1="192" y1="214" x2="192" y2="252" stroke="#3A3A50" stroke-width="4" stroke-linecap="round"/>'
    '<line x1="166" y1="250" x2="218" y2="250" stroke="#3A3A50" stroke-width="3" stroke-linecap="round"/>'
    '<line x1="179" y1="250" x2="192" y2="234" stroke="#3A3A50" stroke-width="2.5" stroke-linecap="round"/>'
    '<line x1="205" y1="250" x2="192" y2="234" stroke="#3A3A50" stroke-width="2.5" stroke-linecap="round"/>'
    '<text x="302" y="158" font-size="22" fill="#F5A623" opacity="0.9" font-family="sans-serif">⚡</text>'
    "</svg>"
)
SOLAR_IDLE = (
    _style
    + '<div class="iw">' + _svg
    + '<div class="it"><h2>Analyse any solar location on Earth</h2><p>Configure your plant in the sidebar and hit RUN ANALYSIS</p></div>'
    + '<div class="is"><div><b>01</b> Set coordinates</div><div><b>02</b> Set capacity</div><div><b>03</b> Add financials (optional)</div><div><b>04</b> Run analysis</div></div>'
    + "</div>"
)

# ════════════════════════════════════════
# TAB 1 — FEASIBILITY
# ════════════════════════════════════════
with t1:
    if not run_btn:
        st.markdown(SOLAR_IDLE, unsafe_allow_html=True)
    else:
        with st.spinner("🛰 Fetching NASA POWER historical data..."):
            res = run_feasibility(
                lat=lat, lon=lon, capacity_kw=capacity_kw,
                cost_inr=cost_inr if cost_inr else 1,
                tariff=tariff   if tariff   else 1,
                model=model, metadata=metadata,
            )
        if not res["available"]:
            st.error("NASA POWER API unavailable. Check internet connection and retry.")
        else:
            cf_c = "#06D6A0" if res["verdict"]=="VIABLE" else "#EF476F" if res["verdict"]=="NOT VIABLE" else "#F5A623"
            v_section = f'{vhtml(res["verdict"])}<div style="font-size:13px;color:#9090A8;margin-top:0.6rem;max-width:580px;">{res["reason"]}</div>'

            st.markdown(f"""
            <div class="sc" style="border-left:4px solid {cf_c};background:linear-gradient(135deg,#1A1A24,#111118);">
              <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.4rem;">ANALYSIS — {lat:.4f}°N, {lon:.4f}°E</div>
                  {v_section}
                </div>
                <div style="text-align:right;">
                  <div style="font-size:11px;letter-spacing:0.1em;color:#6B6B80;">CAPACITY FACTOR</div>
                  <div style="font-size:2.8rem;font-weight:800;color:{cf_c};line-height:1;">{res['capacity_factor']:.2f}%</div>
                  <div style="font-size:11px;color:#6B6B80;margin-top:2px;">target ≥ 18.00%</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            if include_financials:
                mrow([
                    ("Annual Output (kWh)",  f"{res['annual_kwh']:,.2f}",          f"{res['annual_kwh']/capacity_kw:.2f} kWh/kW"),
                    ("Annual Savings (₹)",   f"{res['annual_savings']:,.2f}",       None),
                    ("Payback Period (yrs)", f"{res['payback_years']:.2f}",         "Good" if res['payback_years']<=10 else "Long"),
                    ("Plant Capacity (kW)",  f"{capacity_kw:,}",                   f"{capacity_kw/1000:.3f} MW"),
                ])
            else:
                mrow([
                    ("Annual Output (kWh)",   f"{res['annual_kwh']:,.2f}",          f"{res['annual_kwh']/capacity_kw:.2f} kWh/kW"),
                    ("Capacity Factor (%)",
                    f"{res['capacity_factor']:.4f}",
                    f"{'≥' if res['capacity_factor'] >= 18 else '<'} 18.00% = {'viable' if res['capacity_factor'] >= 18 else 'not viable'}",
                    "normal" if res['capacity_factor'] >= 18 else "inverse"),
                    ("Plant Capacity (kW)",   f"{capacity_kw:,}",                   f"{capacity_kw/1000:.3f} MW"),
                    ("Best Month",            res["monthly_names"][max(res["monthly_kwh"],key=res["monthly_kwh"].get)-1], None),
                ])

            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">ESTIMATED MONTHLY ENERGY OUTPUT (kWh)</div>', unsafe_allow_html=True)

            months = res["monthly_names"]
            vals   = [res["monthly_kwh"][m] for m in range(1,13)]
            colors = ["#F5A623" if v==max(vals) else "#EF476F" if v==min(vals) else "#2A3A5A" for v in vals]

            fig,ax = plt.subplots(figsize=(12,4))
            bars = ax.bar(months, vals, color=colors, width=0.65, zorder=3)
            ax.axhline(np.mean(vals), color="#6B6B80", linewidth=1, linestyle="--",
                       label=f"Avg: {np.mean(vals):,.2f} kWh")
            for bar,val in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                        f"{val:,.0f}", ha="center", va="bottom", fontsize=8, color="#9090A8")
            ax.set_ylabel("kWh / month", fontsize=10)
            ax.legend(fontsize=9)
            ax.set_ylim(0, max(vals)*1.18)
            plt.tight_layout()
            sfig(fig)

            import calendar
            best_m  = max(res["monthly_kwh"], key=res["monthly_kwh"].get)
            worst_m = min(res["monthly_kwh"], key=res["monthly_kwh"].get)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">MONTH-BY-MONTH DETAIL</div>', unsafe_allow_html=True)
            avg_v = np.mean(vals)
            tbl = """<table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;">
              <thead><tr style="border-bottom:1px solid rgba(245,166,35,0.2);">
                <th style="text-align:left;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">MONTH</th>
                <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">OUTPUT (kWh)</th>
                <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">DAILY AVG (kWh)</th>
                <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">VS ANNUAL AVG</th>
                <th style="text-align:center;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">NOTE</th>
              </tr></thead><tbody>"""
            for mn in range(1,13):
                kwh   = res["monthly_kwh"][mn]
                days  = calendar.monthrange(2023,mn)[1]
                daily = kwh/days
                dp    = ((kwh-avg_v)/avg_v)*100
                ds    = f"+{dp:.2f}%" if dp>=0 else f"{dp:.2f}%"
                dc    = "#06D6A0" if dp>=5 else "#EF476F" if dp<=-5 else "#9090A8"
                note  = "Best" if mn==best_m else "Worst" if mn==worst_m else ""
                nc    = "#F5A623" if note=="Best" else "#EF476F" if note=="Worst" else "transparent"
                bg    = "rgba(245,166,35,0.04)" if mn==best_m else "rgba(239,71,111,0.04)" if mn==worst_m else "transparent"
                tbl  += f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:{bg};">
                  <td style="padding:7px 10px;color:#E8E8F0;">{res['monthly_names'][mn-1]}</td>
                  <td style="padding:7px 10px;text-align:right;color:#E8E8F0;">{kwh:,.2f}</td>
                  <td style="padding:7px 10px;text-align:right;color:#9090A8;">{daily:,.2f}</td>
                  <td style="padding:7px 10px;text-align:right;color:{dc};">{ds}</td>
                  <td style="padding:7px 10px;text-align:center;font-size:10px;color:{nc};">{note}</td>
                </tr>"""
            tbl += "</tbody></table>"
            st.markdown(tbl, unsafe_allow_html=True)

            kpk   = res["annual_kwh"]/capacity_kw
            kpk_c = "#06D6A0" if 1200<=kpk<=2000 else "#EF476F"
            st.markdown(f"""
            <div class="sc" style="border-left:3px solid #3A3A50;margin-top:0.75rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
                <div>
                  <div style="font-size:10px;letter-spacing:0.1em;color:#6B6B80;margin-bottom:3px;">OUTPUT INTENSITY</div>
                  <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:500;color:{kpk_c};">{kpk:,.2f} kWh/kW/year</div>
                  <div style="font-size:11px;color:#6B6B80;margin-top:2px;">Benchmark: 1,200 – 2,000 kWh/kW/yr for Indian locations</div>
                </div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#9090A8;text-align:right;">
                  Annual: {res['annual_kwh']:,.2f} kWh<br>
                  Plant: {capacity_kw:,} kW
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if include_financials:
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">CUMULATIVE P&L — 25-YEAR PROJECTION</div>', unsafe_allow_html=True)
                years = list(range(0,26))
                cum   = [res["annual_savings"]*y - cost_inr for y in years]
                fig2,ax2 = plt.subplots(figsize=(12,3.5))
                ax2.fill_between(years,cum,0,where=[v>=0 for v in cum],color="#06D6A0",alpha=0.15,zorder=2)
                ax2.fill_between(years,cum,0,where=[v<0  for v in cum],color="#EF476F",alpha=0.15,zorder=2)
                ax2.plot(years,cum,color="#F5A623",linewidth=1.8,zorder=3)
                ax2.axhline(0,color="#3A3A50",linewidth=0.8)
                ax2.axvline(res["payback_years"],color="#F5A623",linewidth=1,linestyle=":",
                            label=f"Break-even: {res['payback_years']:.2f} yrs")
                ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"₹{x/100000:,.1f}L"))
                ax2.set_xlabel("Year",fontsize=10)
                ax2.set_ylabel("Cumulative P&L (₹)",fontsize=10)
                ax2.legend(fontsize=9)
                plt.tight_layout()
                sfig(fig2)

                st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin:1rem 0 0.5rem;">FINANCIAL SUMMARY</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;">
                  <tbody>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                      <td style="padding:8px 10px;color:#6B6B80;font-size:11px;">Installation Cost</td>
                      <td style="padding:8px 10px;text-align:right;color:#E8E8F0;">₹{cost_inr:,.2f}</td>
                      <td style="padding:8px 10px;text-align:right;color:#9090A8;">₹{cost_inr/capacity_kw:,.2f} / kW</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                      <td style="padding:8px 10px;color:#6B6B80;font-size:11px;">Annual Savings</td>
                      <td style="padding:8px 10px;text-align:right;color:#06D6A0;">₹{res['annual_savings']:,.2f}</td>
                      <td style="padding:8px 10px;text-align:right;color:#9090A8;">₹{res['annual_savings']/12:,.2f} / month</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
                      <td style="padding:8px 10px;color:#6B6B80;font-size:11px;">Payback Period</td>
                      <td style="padding:8px 10px;text-align:right;color:#E8E8F0;">{res['payback_years']:.4f} years</td>
                      <td style="padding:8px 10px;text-align:right;color:#9090A8;">{res['payback_years']*12:.1f} months</td>
                    </tr>
                    <tr>
                      <td style="padding:8px 10px;color:#6B6B80;font-size:11px;">25-Year Net Savings</td>
                      <td style="padding:8px 10px;text-align:right;color:#06D6A0;">₹{(res['annual_savings']*25 - cost_inr):,.2f}</td>
                      <td style="padding:8px 10px;text-align:right;color:#9090A8;">after cost recovery</td>
                    </tr>
                  </tbody>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="sc" style="border-left:3px solid #3A3A50;margin-top:0.5rem;">
                  <div style="font-size:12px;color:#6B6B80;">
                    Enable <b style="color:#F5A623;">Financial Analysis</b> in the sidebar
                    to see payback period, annual savings, and 25-year P&L projection.
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 2 — 7-DAY FORECAST
# ════════════════════════════════════════
with t2:
    if not run_btn:
        st.markdown(SOLAR_IDLE, unsafe_allow_html=True)
    else:
        with st.spinner("🌤 Fetching Open-Meteo 7-day forecast..."):
            fc = run_forecast(lat=lat,lon=lon,capacity_kw=capacity_kw,model=model,metadata=metadata)

        mrow([
            ("7-Day Total (kWh)",   f"{fc['total_7day']:,.2f}",                              None),
            ("Daily Average (kWh)", f"{fc['avg_daily']:,.2f}",                               None),
            ("Peak Day (kWh)",      f"{fc['daily']['predicted_kwh'].max():,.2f}",             None),
            ("Lowest Day (kWh)",    f"{fc['daily']['predicted_kwh'].min():,.2f}",             None),
        ])

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">DAILY OUTPUT FORECAST (kWh)</div>', unsafe_allow_html=True)

        ddf  = fc["daily"]
        lbls = [str(d) for d in ddf["date"]]
        dv   = ddf["predicted_kwh"].tolist()
        maxv = max(dv)
        avg  = np.mean(dv)
        dc   = ["#F5A623" if v==maxv else "#2A3A5A" for v in dv]

        fig3,ax3 = plt.subplots(figsize=(12,4))
        b3 = ax3.bar(lbls,dv,color=dc,width=0.55,zorder=3,edgecolor="none")
        ax3.axhline(avg,color="#6B6B80",linewidth=1,linestyle="--",label=f"Avg: {avg:,.2f} kWh")
        for bar,val in zip(b3,dv):
            ax3.text(bar.get_x()+bar.get_width()/2,bar.get_height()+maxv*0.008,
                     f"{val:,.2f}",ha="center",va="bottom",fontsize=8.5,color="#9090A8")
        ax3.set_ylabel("kWh",fontsize=10)
        ax3.legend(fontsize=9)
        ax3.set_ylim(0,maxv*1.2)
        plt.xticks(rotation=15,ha="right")
        plt.tight_layout()
        sfig(fig3)

        best_date = ddf.loc[ddf["predicted_kwh"].idxmax(),"date"]
        hdf = fc["hourly"].copy()
        hdf["date"] = hdf["time"].dt.date
        bh  = hdf[hdf["date"]==best_date]

        if not bh.empty:
            st.markdown(f'<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-top:1.5rem;margin-bottom:0.5rem;">HOURLY OUTPUT CURVE — {best_date} (BEST DAY)</div>', unsafe_allow_html=True)
            fig4,ax4 = plt.subplots(figsize=(12,3.5))
            hrs = bh["time"].dt.hour
            out = bh["predicted_kw"]
            ax4.fill_between(hrs,out,alpha=0.12,color="#F5A623",zorder=2)
            ax4.plot(hrs,out,color="#F5A623",linewidth=2,zorder=3)
            ax4.scatter(hrs,out,color="#F5A623",s=30,zorder=4,edgecolors="none")
            ph = hrs.iloc[out.values.argmax()]
            pv = out.max()
            ax4.annotate(f"Peak: {pv:,.2f} kW",xy=(ph,pv),xytext=(ph+1,pv*1.04),
                         fontsize=9,color="#9090A8",
                         arrowprops=dict(arrowstyle="->",color="#6B6B80",lw=1))
            ax4.set_xlabel("Hour of day",fontsize=10)
            ax4.set_ylabel("kW",fontsize=10)
            ax4.set_xticks(range(0,24,2))
            plt.tight_layout()
            sfig(fig4)

        st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-top:1.5rem;margin-bottom:0.5rem;">DAY-BY-DAY DETAIL</div>', unsafe_allow_html=True)
        tbl2 = """<table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;">
          <thead><tr style="border-bottom:1px solid rgba(245,166,35,0.2);">
            <th style="text-align:left;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">DATE</th>
            <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">OUTPUT (kWh)</th>
            <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">VS WEEK AVG</th>
            <th style="text-align:right;padding:6px 10px;color:#6B6B80;font-size:10px;letter-spacing:0.08em;font-weight:500;">PEAK kW</th>
          </tr></thead><tbody>"""
        for _,row in ddf.iterrows():
            kwh  = row["predicted_kwh"]
            diff = ((kwh-avg)/avg)*100
            ds   = f"+{diff:.2f}%" if diff>=0 else f"{diff:.2f}%"
            dc2  = "#06D6A0" if diff>=5 else "#EF476F" if diff<=-5 else "#9090A8"
            dh   = hdf[hdf["date"]==row["date"]]
            pk   = dh["predicted_kw"].max() if not dh.empty else 0
            bg   = "rgba(245,166,35,0.04)" if kwh==maxv else "transparent"
            tbl2+= f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.04);background:{bg};">
              <td style="padding:7px 10px;color:#E8E8F0;">{row['date']}</td>
              <td style="padding:7px 10px;text-align:right;color:#E8E8F0;">{kwh:,.2f}</td>
              <td style="padding:7px 10px;text-align:right;color:{dc2};">{ds}</td>
              <td style="padding:7px 10px;text-align:right;color:#9090A8;">{pk:,.2f}</td>
            </tr>"""
        tbl2 += "</tbody></table>"
        st.markdown(tbl2, unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 3 — MODEL DIAGNOSTICS
# ════════════════════════════════════════
with t3:
    m = metadata["metrics"]
    st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin:1rem 0 0.5rem;">MODEL PERFORMANCE METRICS</div>', unsafe_allow_html=True)
    mrow([
        ("R² Score",    f"{m['r2']:.6f}",          "Target > 0.9300"),
        ("nRMSE",       f"{m['nrmse_pct']:.4f}%",   "Target < 10.00%"),
        ("Skill Score", f"{m['skill_score']:.4f}%",  "vs persistence"),
        ("Model",       metadata["model_name"],      None),
    ])

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
        fig_fi = plot_feature_importance(model)
        if fig_fi: sfig(fig_fi)
        else: st.info("Feature importance not available for this model type.")

    with c2:
        st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin-bottom:0.5rem;">TRAINING INFO</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc sc-a" style="margin-top:0;">
          <table style="font-family:'JetBrains Mono',monospace;font-size:12px;width:100%;border-collapse:collapse;">
            <tr><td style="color:#6B6B80;padding:4px 0;">Plant capacity</td><td style="text-align:right;color:#E8E8F0;">{metadata['plant_capacity_kw']:,.2f} kW</td></tr>
            <tr><td style="color:#6B6B80;padding:4px 0;">Features used</td> <td style="text-align:right;color:#E8E8F0;">{len(metadata['features'])}</td></tr>
            <tr><td style="color:#6B6B80;padding:4px 0;">Target variable</td><td style="text-align:right;color:#E8E8F0;">{metadata['target']}</td></tr>
          </table>
        </div>
        <div class="sc" style="border-left:3px solid #3A3A50;margin-top:0.5rem;">
          <div style="font-size:10px;letter-spacing:0.1em;color:#6B6B80;margin-bottom:6px;">FEATURES</div>
          {''.join([f'<span style="display:inline-block;background:#0A0A0F;border:1px solid rgba(245,166,35,0.15);border-radius:4px;padding:2px 8px;margin:2px;font-family:JetBrains Mono,monospace;font-size:10px;color:#F5A623;">{f}</span>' for f in metadata['features']])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px;letter-spacing:0.12em;color:#6B6B80;margin:1.5rem 0 0.5rem;">MODEL COMPARISON</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;">
      <thead><tr style="border-bottom:1px solid rgba(245,166,35,0.2);">
        <th style="text-align:left;padding:0.6rem 1rem;color:#6B6B80;font-size:10px;letter-spacing:0.1em;font-weight:500;">MODEL</th>
        <th style="text-align:right;padding:0.6rem 1rem;color:#6B6B80;font-size:10px;letter-spacing:0.1em;font-weight:500;">R²</th>
        <th style="text-align:right;padding:0.6rem 1rem;color:#6B6B80;font-size:10px;letter-spacing:0.1em;font-weight:500;">nRMSE</th>
        <th style="text-align:right;padding:0.6rem 1rem;color:#6B6B80;font-size:10px;letter-spacing:0.1em;font-weight:500;">SKILL SCORE</th>
        <th style="text-align:right;padding:0.6rem 1rem;color:#6B6B80;font-size:10px;letter-spacing:0.1em;font-weight:500;">STATUS</th>
      </tr></thead>
      <tbody>
        <tr style="border-bottom:1px solid rgba(255,255,255,0.04);">
          <td style="padding:0.6rem 1rem;color:#9090A8;">Linear Regression</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#E8E8F0;">0.872800</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#E8E8F0;">8.6600%</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#E8E8F0;">57.3000%</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#6B6B80;font-size:10px;">Baseline</td>
        </tr>
        <tr>
          <td style="padding:0.6rem 1rem;color:#F5A623;font-weight:600;">XGBoost</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#06D6A0;">{m['r2']:.6f}</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#06D6A0;">{m['nrmse_pct']:.4f}%</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#06D6A0;">{m['skill_score']:.4f}%</td>
          <td style="padding:0.6rem 1rem;text-align:right;color:#06D6A0;font-size:10px;">Selected</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sc" style="margin-top:1.5rem;border-left:3px solid #3A3A50;">
      <div style="font-size:10px;letter-spacing:0.1em;color:#6B6B80;margin-bottom:0.4rem;">DATASET NOTE</div>
      <div style="font-size:13px;color:#9090A8;line-height:1.8;">
        Training covers May–June 2020 only (34 days, 2 Indian plants, 3,507 rows after inverter aggregation).
        R² {m['r2']:.6f} and nRMSE {m['nrmse_pct']:.4f}% are strong given this constraint.
        Key fix: inverter aggregation resolved data consistency and lifted R² from 0.760000 → {m['r2']:.6f}.
      </div>
    </div>
    """, unsafe_allow_html=True)

# CHANGE 3: Removed "ML COURSE PROJECT" from footer
st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid rgba(245,166,35,0.08);
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
  <div style="font-size:11px;color:#6B6B80;letter-spacing:0.08em;">SOLARSENSE · 2024–25</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#3A3A50;">
    Kanakamurthy H 
  </div>
</div>
""", unsafe_allow_html=True)