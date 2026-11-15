"""
AI analyst: Claude with tool use over SolarSense's own engines.

Claude never computes energy figures itself. It calls these tools and may
only quote numbers they return:
    find_city            coordinates for an Indian city name
    analyse_site         P50/P90 energy, CUF, losses, climate, verdict
    compare_sites        analyse_site for several sites, side by side
    atlas_lookup         36-city ranking (yield, CUF, monsoon dip, tilt)
    forecast_production  7-day P10/P50/P90 production forecast

Optional: needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY, or
an `ant auth login` profile). The app shows setup instructions otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from solarsense.config import ATLAS_PATH, INDIA_LOCATIONS_PATH

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 8

SYSTEM_PROMPT = """You are SolarSense Analyst, a solar energy-yield analyst for India.

You have tools backed by a validated engine: pvlib physics on 10 years of hourly ERA5 weather, calibrated on real metered PV plants, with P50/P90 uncertainty.

Rules:
- Every number you state (energy, yield, CUF, losses, forecast) must come from a tool result in this conversation. Never estimate or recall such numbers yourself. If a tool fails, say so.
- Look up coordinates with find_city before analysing a named Indian city.
- If the user gives no plant details, use the tool defaults (DC/AC 1.25, fixed tilt) and say so.
- SolarSense covers energy only. If asked about costs, returns or tariffs, say that is outside its scope.
- Explain P50 (expected) vs P90 (exceeded 9 years in 10) when it matters for the decision.
- Be concise: lead with the answer, then a short table or bullets with the key figures, then caveats.
- Energy in MWh or GWh."""

TOOLS = [
    {
        "name": "find_city",
        "description": "Return latitude/longitude for an Indian city from SolarSense's 36-city list. "
                       "Use before analysing a named city. Returns close matches if the exact name is not found.",
        "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
                         "additionalProperties": False},
    },
    {
        "name": "analyse_site",
        "description": "Full feasibility for one site: P50 and P90 annual energy, specific yield, CUF, "
                       "performance ratio, monthly profile, climate summary and a VIABLE/MARGINAL/NOT VIABLE "
                       "energy verdict. "
                       "Takes ~3-20 s (fetches 10 years of hourly weather on first use).",
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"},
                "capacity_kwp": {"type": "number", "description": "DC capacity in kWp (1 MW = 1000 kWp)"},
                "dc_ac_ratio": {"type": "number", "description": "default 1.25"},
                "tracking": {"type": "string", "enum": ["fixed", "single_axis"]},
            },
            "required": ["latitude", "longitude", "capacity_kwp"],
            "additionalProperties": False,
        },
    },
    {
        "name": "compare_sites",
        "description": "Run analyse_site for 2-6 sites with identical plant inputs and return a "
                       "comparison table. Use for 'which is better' questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sites": {"type": "array", "minItems": 2, "maxItems": 6, "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "latitude": {"type": "number"},
                                   "longitude": {"type": "number"}},
                    "required": ["name", "latitude", "longitude"], "additionalProperties": False}},
                "capacity_kwp": {"type": "number"},
                "dc_ac_ratio": {"type": "number"},
                "tracking": {"type": "string", "enum": ["fixed", "single_axis"]},
            },
            "required": ["sites", "capacity_kwp"],
            "additionalProperties": False,
        },
    },
    {
        "name": "atlas_lookup",
        "description": "Ranking of 36 Indian reference cities by calibrated specific yield (kWh/kWp/yr) for a "
                       "standard 1 kWp fixed-tilt plant, with CUF, best/worst month, monsoon dip and optimal tilt. "
                       "Optionally filter by region or state.",
        "input_schema": {
            "type": "object",
            "properties": {"region": {"type": "string"}, "state": {"type": "string"},
                           "top_n": {"type": "integer", "minimum": 1, "maximum": 36}},
            "additionalProperties": False,
        },
    },
    {
        "name": "forecast_production",
        "description": "7-day hourly production forecast for a plant, summarised per day as P10/P50/P90 kWh "
                       "(80% calibrated band) with cloud cover.",
        "input_schema": {
            "type": "object",
            "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"},
                           "capacity_kwp": {"type": "number"}, "dc_ac_ratio": {"type": "number"}},
            "required": ["latitude", "longitude", "capacity_kwp"],
            "additionalProperties": False,
        },
    },
]


# ─────────────────────────────────────────────────────────────
# AVAILABILITY
# ─────────────────────────────────────────────────────────────

def is_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


def setup_instructions() -> str:
    return (
        "### 🤖 AI Analyst (optional)\n"
        "Ask questions like *“Which yields more for 5 MW, Bikaner or Coimbatore?”*. Claude answers by "
        "calling SolarSense's physics, forecast and atlas tools, and only quotes numbers they return.\n\n"
        "**To enable:**\n"
        "1. `pip install anthropic`\n"
        "2. Set credentials: `export ANTHROPIC_API_KEY=...` (or run `ant auth login`)\n"
        "3. Restart `streamlit run app/app.py`\n\n"
        f"Model: `{MODEL}`. Each question makes a few API calls (billed to your Anthropic account)."
    )


# ─────────────────────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────

def _system(args: dict):
    from solarsense.physics import PVSystem
    return PVSystem(capacity_kwp=float(args["capacity_kwp"]), dc_ac_ratio=float(args.get("dc_ac_ratio", 1.25)),
                    tracking=args.get("tracking", "fixed"))


def _site_summary(name: str, lat: float, lon: float, args: dict) -> dict:
    from solarsense.feasibility import run_feasibility
    r = run_feasibility(lat, lon, _system(args))
    out = {
        "site": name, "latitude": lat, "longitude": lon,
        "capacity_kwp": r.system.capacity_kwp, "dc_ac_ratio": r.system.dc_ac_ratio, "tilt_deg": round(r.tilt, 1),
        "weather": f"{r.weather_source} {r.years[0]}-{r.years[-1]}",
        "p50_mwh_per_year": round(r.p50_kwh / 1e3, 1),
        "p90_mwh_per_year": round(r.p90_kwh / 1e3, 1),
        "specific_yield_p50_kwh_per_kwp": round(r.specific_yield_p50),
        "cuf_ac_pct": round(r.cuf_ac_pct, 2),
        "performance_ratio": round(float(r.monthly["performance_ratio"].mean()), 3),
        "uncertainty_pct_1sigma": {k: round(v, 2) for k, v in r.uncertainty.items()},
        "climate": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.climate.items()},
        "monthly_specific_yield": dict(zip(r.monthly["month"], r.monthly["specific_yield"].round(1), strict=True)),
        "verdict": r.verdict,
    }
    return out


def find_city(args: dict) -> dict:
    locs = pd.read_csv(INDIA_LOCATIONS_PATH)
    q = args["name"].strip().lower()
    exact = locs[(locs["city"].str.lower() == q) | (locs["location_id"] == q.replace(" ", "_"))]
    if len(exact):
        r = exact.iloc[0]
        return {"city": r.city, "state": r.state, "latitude": r.latitude, "longitude": r.longitude}
    close = locs[locs["city"].str.lower().str.contains(q[:4])]
    return {"error": f"'{args['name']}' is not in the 36-city list",
            "close_matches": close[["city", "state", "latitude", "longitude"]].to_dict(orient="records"),
            "hint": "If the user knows the coordinates, call analyse_site with them directly."}


def analyse_site(args: dict) -> dict:
    return _site_summary("site", float(args["latitude"]), float(args["longitude"]), args)


def compare_sites(args: dict) -> dict:
    rows = [_site_summary(s["name"], float(s["latitude"]), float(s["longitude"]), args) for s in args["sites"]]
    table = [{"site": r["site"], "p50_mwh": r["p50_mwh_per_year"], "p90_mwh": r["p90_mwh_per_year"],
              "specific_yield": r["specific_yield_p50_kwh_per_kwp"], "cuf_ac_pct": r["cuf_ac_pct"],
              "monsoon_dip_pct": r["climate"]["monsoon_dip_pct"], "verdict": r["verdict"]["label"]}
             for r in rows]
    return {"comparison": table, "details": rows}


def atlas_lookup(args: dict) -> dict:
    if not ATLAS_PATH.exists():
        return {"error": "atlas not built - run scripts/build_india_atlas.py"}
    a = pd.read_csv(ATLAS_PATH)
    if args.get("region"):
        a = a[a["region"].str.lower() == args["region"].lower()]
    if args.get("state"):
        a = a[a["state"].str.lower() == args["state"].lower()]
    a = a.head(int(args.get("top_n", 36)))
    cols = ["rank", "city", "state", "region", "specific_yield_p50", "cuf_ac_pct", "best_month", "worst_month",
            "monsoon_dip_pct", "optimal_tilt"]
    return {"basis": "1 kWp fixed tilt, DC/AC 1.25, default Indian losses, ERA5 2020-2022, calibrated",
            "rows": a[cols].to_dict(orient="records")}


def forecast_production(args: dict) -> dict:
    from solarsense.forecast import run_forecast
    fc = run_forecast(float(args["latitude"]), float(args["longitude"]), _system(args))
    d = fc.daily.copy()
    d["date"] = d["date"].astype(str)
    return {"days": d.round(1).to_dict(orient="records"), "totals_kwh": {k: round(v) for k, v in fc.total.items()},
            "band": "P10-P90 calibrated to cover 80% of outcomes", "notes": fc.notes}


HANDLERS = {"find_city": find_city, "analyse_site": analyse_site, "compare_sites": compare_sites,
            "atlas_lookup": atlas_lookup, "forecast_production": forecast_production}


def run_tool(name: str, args: dict) -> tuple[str, bool]:
    """Execute a tool; returns (json_text, is_error)."""
    try:
        if name not in HANDLERS:
            raise ValueError(f"unknown tool {name}")
        return json.dumps(HANDLERS[name](args), default=float), False
    except Exception as e:  # noqa: BLE001 - surface any failure to the model as a tool error
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True


# ─────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────

def ask(chat: list[dict]) -> tuple[str, list[dict]]:
    """
    chat: [{"role": "user"|"assistant", "content": str}, ...] ending with a user turn.
    Returns (answer_text, trace) where trace lists the tool calls made.
    """
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": m["role"], "content": m["content"]} for m in chat]
    trace: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                output_config={"effort": "medium"},
                # Server-side fallback: if a safety classifier declines, the API
                # re-runs the request on a suitable fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError:
            return "Anthropic credentials were rejected - check ANTHROPIC_API_KEY.", trace
        except anthropic.RateLimitError:
            return "Rate limited by the Anthropic API - try again in a minute.", trace
        except anthropic.APIStatusError as e:
            return f"Anthropic API error ({e.status_code}): {e.message}", trace
        except anthropic.APIConnectionError:
            return "Could not reach the Anthropic API - check the network connection.", trace

        if response.stop_reason == "refusal":
            return "The model declined this request.", trace
        if response.stop_reason == "max_tokens":
            return "The answer was cut off (max_tokens). Please ask a narrower question.", trace

        messages.append({"role": "assistant", "content": response.content})
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if response.stop_reason == "pause_turn":
            continue
        if not tool_uses:
            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            return text or "(no answer)", trace

        results = []
        for tu in tool_uses:
            out, is_error = run_tool(tu.name, dict(tu.input))
            trace.append({"tool": tu.name, "input": dict(tu.input), "error": is_error})
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": is_error})
        messages.append({"role": "user", "content": results})

    return "Stopped after too many tool rounds - please ask a more specific question.", trace
