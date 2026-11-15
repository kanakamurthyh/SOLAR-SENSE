"""
Train and validate the ML models on real plant data.

Real-plant output differs from physics in two separable ways:

    y(t) = k_plant  x  aging(t)  x  z(t)

    k_plant  a constant per plant: nameplate tolerance, soiling regime,
             wiring, inverter - things weather features cannot explain.
             Its spread across plants IS the energy uncertainty for a new,
             unmeasured plant, so it feeds P90 in feasibility.
    aging    known degradation since commissioning (removed so every plant
             is compared as new).
    z(t)     the hourly shape: what ML can learn and transfer between plants
             (clearness, low light, temperature, clipping, NWP error).

For each weather source ("forecast" = archived NWP forecasts, "hindcast" =
ERA5 reanalysis) this script:
    1. computes k_plant and z(t) per plant,
    2. runs leave-one-plant-out CV of the quantile model on z, against
       physics (pvlib + default loss chain) and 24 h persistence,
    3. trains the final model on all plants and writes a model card.

Outputs:
    models/{forecast,hindcast}_model.ubj + _model_card.json
    reports/ml_cv_metrics.csv, reports/cv_predictions.parquet

Run from project root:
    python3 -m training.train
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from solarsense.config import REPORTS_DIR, TRAINING_SET_PATH
from solarsense.ml import FEATURES, fit, score
from training.plants import DEGRADATION_PCT_PER_YEAR, PLANTS


def normalise(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = d.copy()
    aging = np.ones(len(d))
    for pid, spec in PLANTS.items():
        m = (d["plant"] == pid).to_numpy()
        if not m.any():
            continue
        age = (d.loc[m, "time"] - pd.Timestamp(spec["commissioned"], tz="UTC")).dt.days / 365.25
        rate = DEGRADATION_PCT_PER_YEAR[spec["tech"]] / 100
        aging[m] = (1 - rate) ** age.clip(lower=0).to_numpy()
    d["aging"] = aging
    baseline = d["phys_lossy_pu"] * d["aging"]
    k = d.assign(b=baseline).groupby("plant").apply(lambda g: g["target_pu"].sum() / g["b"].sum())
    d["k_plant"] = d["plant"].map(k)
    d["target_norm"] = d["target_pu"] / (d["k_plant"] * d["aging"])
    return d, k


def persistence_24h(d: pd.DataFrame, col: str) -> np.ndarray:
    s = d.set_index(["plant", "time"])[col]
    prev = s.copy()
    prev.index = pd.MultiIndex.from_arrays([prev.index.get_level_values(0),
                                            prev.index.get_level_values(1) + pd.Timedelta(hours=24)])
    return prev.reindex(s.index).to_numpy()


def cross_validate(d: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, preds = [], []
    for held in sorted(d["plant"].unique()):
        train, test = d[d["plant"] != held], d[d["plant"] == held].copy()
        model = fit(train[FEATURES], train["target_norm"], name=f"cv_{source}")
        test[["p10", "p50", "p90"]] = model.predict(test[FEATURES]).to_numpy()
        preds.append(test)
        z = test["target_norm"]
        entries = {
            "ml_p50": score(z, test["p50"], test["p10"], test["p90"]),
            "physics": score(z, test["phys_lossy_pu"]),
        }
        if source == "forecast":
            entries["persistence_24h"] = score(z, pd.Series(test["persist"].to_numpy(), index=test.index))
        for method, s in entries.items():
            rows.append({"source": source, "held_out_plant": held, "method": method, **s})
        ml, ph = entries["ml_p50"], entries["physics"]
        extra = ""
        if source == "forecast":
            extra = f" | persistence {entries['persistence_24h']['nrmse_pct']:5.2f}%"
        print(f"  {held:26s} ML {ml['nrmse_pct']:5.2f}%  cover {ml['coverage_p10_p90_pct']:4.0f}%"
              f" | physics {ph['nrmse_pct']:5.2f}%{extra}", flush=True)
    return pd.DataFrame(rows), pd.concat(preds)


HOURLY_CSI_BINS = [0.0, 0.3, 0.6, 0.8, 0.9, 1.6]
DAILY_CSI_BINS = [0.0, 0.5, 0.75, 0.9, 1.6]
GRID = np.round(np.arange(0.05, 8.01, 0.05), 2)


def _min_scale(y, c, lo_w, hi_w, target):
    for g in GRID:
        if np.mean((y >= c - g * lo_w) & (y <= c + g * hi_w)) >= target:
            return float(g)
    return float(GRID[-1])


def _mondrian(frame: pd.DataFrame, bins: list, target: float, min_n: int) -> tuple[dict, float, float]:
    """Per clear-sky-index bin conformal scales (Mondrian conformal prediction)."""
    y, c, lo, hi = (frame[k].to_numpy() for k in ("y", "c", "lo_w", "hi_w"))
    global_s = _min_scale(y, c, lo, hi, target)
    cats = pd.cut(frame["csi"], bins, labels=False, include_lowest=True).fillna(0).astype(int).to_numpy()
    scales = {}
    for b in range(len(bins) - 1):
        m = cats == b
        scales[str(b)] = _min_scale(y[m], c[m], lo[m], hi[m], target) if m.sum() >= min_n else global_s
    s_row = np.array([scales[str(b)] for b in cats])
    cov = float(np.mean((y >= c - s_row * lo) & (y <= c + s_row * hi)))
    return scales, global_s, cov


def calibrate_bands(preds: pd.DataFrame, target: float = 0.80) -> dict:
    """
    Conformal calibration of the uncertainty band around PHYSICS (the P50 used
    in the app): band = physics -/+ s * (ML quantile half-widths).

    Mondrian (group-conditional) conformal: a separate scale s per bin of the
    forecast clear-sky index, so clear forecasts get tight bands and cloudy
    ones wide bands, each covering `target` of out-of-plant outcomes.
    Daily totals are calibrated separately (hourly errors partly cancel).
    """
    h = pd.DataFrame({
        "y": preds["target_norm"].to_numpy(), "c": preds["phys_lossy_pu"].to_numpy(),
        "lo_w": (preds["p50"] - preds["p10"]).clip(lower=0).to_numpy(),
        "hi_w": (preds["p90"] - preds["p50"]).clip(lower=0).to_numpy(),
        "csi": preds["clearsky_index"].fillna(0).to_numpy(),
    })
    hourly_scales, hourly_global, hourly_cov = _mondrian(h, HOURLY_CSI_BINS, target, min_n=300)

    d = h.assign(plant=preds["plant"].to_numpy(), day=preds["time"].dt.floor("D").to_numpy(),
                 csi_w=h["csi"] * h["c"])
    daily = d.groupby(["plant", "day"]).agg(y=("y", "sum"), c=("c", "sum"), lo_w=("lo_w", "sum"),
                                            hi_w=("hi_w", "sum"), csi_w=("csi_w", "sum"))
    daily["csi"] = (daily["csi_w"] / daily["c"].where(daily["c"] > 0)).fillna(0)
    daily_scales, daily_global, daily_cov = _mondrian(daily, DAILY_CSI_BINS, target, min_n=40)

    return {
        "method": "Mondrian conformal on clear-sky-index bins, leave-one-plant-out predictions",
        "hourly_csi_bins": HOURLY_CSI_BINS, "hourly_scales": hourly_scales,
        "band_scale": hourly_global, "hourly_coverage_pct": round(hourly_cov * 100, 1),
        "daily_csi_bins": DAILY_CSI_BINS, "daily_scales": daily_scales,
        "daily_band_scale": daily_global, "daily_coverage_pct": round(daily_cov * 100, 1),
        "uncalibrated_coverage_pct": round(float(np.mean((h.y >= h.c - h.lo_w) & (h.y <= h.c + h.hi_w))) * 100, 1),
    }


def main():
    df = pd.read_parquet(TRAINING_SET_PATH)
    df = df[df["target_pu"].notna()]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    all_metrics, all_preds = [], []
    for source in ("forecast", "hindcast"):
        d, k = normalise(df[df["source"] == source])
        if d["plant"].nunique() < 2:
            continue
        if source == "forecast":
            d["persist"] = persistence_24h(d, "target_norm")
        print(f"\n=== {source}: {len(d):,} daylight hours, {d['plant'].nunique()} plants ===")
        print("  plant energy factor k (measured / physics, age-corrected):",
              {p: round(v, 3) for p, v in k.items()})
        metrics, preds = cross_validate(d, source)
        all_metrics.append(metrics)
        all_preds.append(preds)

        summary = {}
        for method, g in metrics.groupby("method"):
            summary[method] = {"mean_nrmse_pct": round(g["nrmse_pct"].mean(), 2),
                               "mean_nmae_pct": round(g["nmae_pct"].mean(), 2)}
            if g["coverage_p10_p90_pct"].notna().any():
                summary[method]["mean_coverage_p10_p90_pct"] = round(g["coverage_p10_p90_pct"].mean(), 1)
        ml_n, ph_n = summary["ml_p50"]["mean_nrmse_pct"], summary["physics"]["mean_nrmse_pct"]
        card = {
            "source": source,
            "training_plants": sorted(d["plant"].unique()),
            "training_period": [str(d["time"].min()), str(d["time"].max())],
            "target": "AC per kWp, normalised by plant energy factor k and age (new plant, default losses)",
            "cv": "leave-one-plant-out",
            "cv_summary": summary,
            "cv_per_plant": metrics.to_dict(orient="records"),
            "skill_vs_physics_pct": round((1 - ml_n / ph_n) * 100, 1),
            "plant_energy_factor_k": {p: round(float(v), 4) for p, v in k.items()},
            "k_mean": round(float(k.mean()), 4),
            "k_median": round(float(k.median()), 4),
            "k_std": round(float(k.std(ddof=1)), 4),
        }
        card["band_calibration"] = calibrate_bands(preds)
        print(f"  conformal band calibration: {card['band_calibration']}")
        if "persistence_24h" in summary:
            card["skill_vs_persistence_pct"] = round((1 - ml_n / summary["persistence_24h"]["mean_nrmse_pct"]) * 100, 1)
            card["physics_skill_vs_persistence_pct"] = round(
                (1 - ph_n / summary["persistence_24h"]["mean_nrmse_pct"]) * 100, 1)
        final = fit(d[FEATURES], d["target_norm"], name=source, card=card)
        final.save()
        print(f"  summary: {json.dumps(summary)}")
        print(f"  ML skill vs physics: {card['skill_vs_physics_pct']:+.1f}%"
              + (f" | vs persistence: {card['skill_vs_persistence_pct']:+.1f}%" if "skill_vs_persistence_pct" in card else ""))
        print(f"  k: mean {card['k_mean']:.3f} median {card['k_median']:.3f} std {card['k_std']:.3f}")

    REPORTS_DIR.mkdir(exist_ok=True)
    pd.concat(all_metrics).to_csv(REPORTS_DIR / "ml_cv_metrics.csv", index=False)
    keep = ["time", "plant", "source", "target_norm", "phys_lossy_pu", "p10", "p50", "p90",
            "solar_elevation", "clearsky_index"]
    pd.concat(all_preds)[keep].to_parquet(REPORTS_DIR / "cv_predictions.parquet", index=False)
    print(f"\nreports written to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
