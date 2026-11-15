"""
ML layer: quantile XGBoost that corrects the physics model towards what real
plants actually produce, and quantifies uncertainty (P10 / P50 / P90).

Two models share this code:
    forecast  : features from NWP *forecast* weather  -> measured output
                (used by the 7-day forecast)
    hindcast  : features from ERA5 *reanalysis* weather -> measured output
                (used to validate / calibrate feasibility)

Design rules:
    * Target is AC power per kWp DC, so plants of any size share one model.
    * No latitude, longitude, month or plant id among the features: the model
      is trained on US plants and applied in India, so it may only learn
      physically transferable relations (irradiance, geometry, temperature,
      clearness, clipping) - not memorise a climate.
    * Physics output is the most important feature; ML learns the residual
      (soiling, low-light, thermal, inverter behaviour, weather-source bias).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from solarsense.config import MODELS_DIR

QUANTILES = (0.1, 0.5, 0.9)

FEATURES = [
    "phys_ac_pu",          # physics AC per kWp, no system losses, clipped at the plant's AC limit
    "phys_dc_pu",          # physics DC per kWp after temperature, before losses
    "poa_kwm2",            # plane-of-array irradiance
    "ghi_kwm2",
    "dni_kwm2",
    "diffuse_fraction",    # dhi / ghi
    "clearsky_index",
    "solar_elevation",
    "cos_aoi",
    "temp_air",
    "temp_cell",
    "wind_speed",
    "cloud_cover",
    "relative_humidity",
    "ac_limit_pu",         # 1 / dc_ac_ratio
]

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=np.array(QUANTILES),
    n_estimators=600,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


def make_features(weather: pd.DataFrame, sim_ideal: pd.DataFrame, capacity_kwp: float,
                  dc_ac_ratio: float) -> pd.DataFrame:
    """
    Build the ML feature frame.

    sim_ideal must be `physics.simulate(..., losses=False, clip=True)` for the
    plant, so that `phys_ac_pu` contains clipping but no assumed losses.
    """
    ghi = weather["ghi"]
    f = pd.DataFrame(index=weather.index)
    f["phys_ac_pu"] = sim_ideal["ac"] / capacity_kwp
    f["phys_dc_pu"] = sim_ideal["dc_gross"] / capacity_kwp
    f["poa_kwm2"] = sim_ideal["poa_global"] / 1000.0
    f["ghi_kwm2"] = ghi / 1000.0
    f["dni_kwm2"] = weather["dni"] / 1000.0
    f["diffuse_fraction"] = (weather["dhi"] / ghi.where(ghi > 10)).clip(0, 1)
    f["clearsky_index"] = sim_ideal["clearsky_index"]
    f["solar_elevation"] = sim_ideal["solar_elevation"]
    f["cos_aoi"] = np.cos(np.radians(sim_ideal["aoi"].clip(0, 90)))
    f["temp_air"] = weather["temp_air"]
    f["temp_cell"] = sim_ideal["temp_cell"]
    f["wind_speed"] = weather["wind_speed"]
    f["cloud_cover"] = weather["cloud_cover"] if "cloud_cover" in weather else np.nan
    f["relative_humidity"] = weather["relative_humidity"] if "relative_humidity" in weather else np.nan
    f["ac_limit_pu"] = 1.0 / dc_ac_ratio
    return f[FEATURES].astype("float32")


def daylight_mask(features: pd.DataFrame) -> pd.Series:
    return features["solar_elevation"] > 0


# ─────────────────────────────────────────────────────────────
# MODEL WRAPPER
# ─────────────────────────────────────────────────────────────

@dataclass
class QuantileModel:
    booster: xgb.XGBRegressor
    name: str
    card: dict

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return P10/P50/P90 AC per kWp, zero at night, monotone across quantiles."""
        out = pd.DataFrame(0.0, index=features.index, columns=["p10", "p50", "p90"])
        day = daylight_mask(features)
        if day.any():
            raw = np.asarray(self.booster.predict(features.loc[day, FEATURES]))
            raw = np.sort(np.clip(raw, 0, None), axis=1)
            out.loc[day, ["p10", "p50", "p90"]] = raw
        # never exceed the plant AC limit
        cap = features["ac_limit_pu"].to_numpy()[:, None]
        out[:] = np.minimum(out.to_numpy(), cap)
        return out

    def save(self, directory: Path = MODELS_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(directory / f"{self.name}_model.ubj")
        (directory / f"{self.name}_model_card.json").write_text(json.dumps(self.card, indent=2, default=str))

    @classmethod
    def load(cls, name: str, directory: Path = MODELS_DIR) -> QuantileModel:
        path = directory / f"{name}_model.ubj"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python3 -m training.train`")
        booster = xgb.XGBRegressor()
        booster.load_model(path)
        card_path = directory / f"{name}_model_card.json"
        card = json.loads(card_path.read_text()) if card_path.exists() else {}
        if card.get("features") and card["features"] != FEATURES:
            raise ValueError(f"{name} model was trained with different features - retrain")
        return cls(booster, name, card)


def fit(features: pd.DataFrame, target_pu: pd.Series, name: str, card: dict | None = None,
        params: dict | None = None) -> QuantileModel:
    day = daylight_mask(features) & target_pu.notna()
    booster = xgb.XGBRegressor(**(params or XGB_PARAMS))
    booster.fit(features.loc[day, FEATURES], target_pu[day])
    card = dict(card or {})
    card.update(features=FEATURES, quantiles=list(QUANTILES), n_train_rows=int(day.sum()))
    return QuantileModel(booster, name, card)


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

def pinball(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def score(y_pu: pd.Series, pred_pu: pd.Series, p10: pd.Series | None = None,
          p90: pd.Series | None = None) -> dict:
    """Metrics on daylight hours, all relative to plant capacity (per-unit)."""
    m = y_pu.notna() & pred_pu.notna()
    y, p = y_pu[m].to_numpy(), pred_pu[m].to_numpy()
    out = {
        "n_hours": int(m.sum()),
        "nrmse_pct": float(np.sqrt(np.mean((p - y) ** 2)) * 100),
        "nmae_pct": float(np.mean(np.abs(p - y)) * 100),
        "mbe_pct_of_energy": float((p.sum() / y.sum() - 1) * 100) if y.sum() > 0 else np.nan,
    }
    if p10 is not None and p90 is not None:
        lo, hi = p10[m].to_numpy(), p90[m].to_numpy()
        out["coverage_p10_p90_pct"] = float(np.mean((y >= lo) & (y <= hi)) * 100)
        out["pinball_mean"] = float(np.mean([pinball(y, lo, 0.1), pinball(y, p, 0.5), pinball(y, hi, 0.9)]))
    return out
