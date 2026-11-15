"""
Real-plant calibration read from the model cards written by training/train.py.

    k        mean ratio measured / physics across real plants (age-corrected).
             < 1 means physics with the default loss chain is optimistic.
    k_std    plant-to-plant spread of k: the energy uncertainty for a new,
             unmeasured plant.
    band     conformal scale factors that make ML quantile bands cover 80 %
             of real outcomes on plants the model never saw.

If a card is missing (models not trained), neutral values are returned and
`available` is False so the app can say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from solarsense.config import FALLBACK_MODEL_UNCERTAINTY_PCT, MIN_FORECAST_ERROR_GROWTH_PER_DAY, MODELS_DIR


@dataclass(frozen=True)
class Calibration:
    available: bool
    k: float = 1.0
    k_std: float = FALLBACK_MODEL_UNCERTAINTY_PCT / 100
    band_scale: float = 1.0
    daily_band_scale: float = 1.0
    hourly_bins: tuple = ()
    hourly_scales: dict = field(default_factory=dict)
    daily_bins: tuple = ()
    daily_scales: dict = field(default_factory=dict)
    lead_ratio: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)

    def lead_factor(self, lead_day: int) -> float:
        measured = float(self.lead_ratio.get(str(lead_day), self.lead_ratio.get(lead_day, 1.0)))
        return max(measured, 1.0 + MIN_FORECAST_ERROR_GROWTH_PER_DAY * lead_day)

    @staticmethod
    def _scales(csi: pd.Series, bins: tuple, scales: dict, default: float) -> np.ndarray:
        if not bins or not scales:
            return np.full(len(csi), default)
        cats = pd.cut(csi.fillna(0), list(bins), labels=False, include_lowest=True).fillna(0).astype(int)
        return np.array([scales.get(str(c), default) for c in cats])

    def hourly_scale(self, csi: pd.Series) -> np.ndarray:
        """Conformal band scale per hour from the forecast clear-sky index."""
        return self._scales(csi, self.hourly_bins, self.hourly_scales, self.band_scale)

    def daily_scale(self, csi: pd.Series) -> np.ndarray:
        return self._scales(csi, self.daily_bins, self.daily_scales, self.daily_band_scale)


@lru_cache(maxsize=4)
def load_calibration(name: str) -> Calibration:
    path = MODELS_DIR / f"{name}_model_card.json"
    if not path.exists():
        return Calibration(available=False)
    card = json.loads(path.read_text())
    bc = card.get("band_calibration", {})
    return Calibration(
        available=True,
        k=float(card.get("k_mean", 1.0)),
        k_std=float(card.get("k_std", FALLBACK_MODEL_UNCERTAINTY_PCT / 100)),
        band_scale=float(bc.get("band_scale", 1.0)),
        daily_band_scale=float(bc.get("daily_band_scale", 1.0)),
        hourly_bins=tuple(bc.get("hourly_csi_bins", ())),
        hourly_scales=dict(bc.get("hourly_scales", {})),
        daily_bins=tuple(bc.get("daily_csi_bins", ())),
        daily_scales=dict(bc.get("daily_scales", {})),
        lead_ratio={str(k): v for k, v in card.get("lead_day_error_ratio", {}).items()},
        card=card,
    )
