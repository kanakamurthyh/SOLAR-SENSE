import numpy as np
import pandas as pd
import pytest

from solarsense.ml import FEATURES, QuantileModel, fit, make_features, score
from solarsense.physics import PVSystem, simulate


def _features(weather, dc_ac=1.25):
    s = PVSystem(capacity_kwp=1, dc_ac_ratio=dc_ac)
    ideal = simulate(weather, s, losses=False, clip=True)
    return make_features(weather, ideal, 1.0, dc_ac), ideal


def test_feature_frame_contract(jaipur_week):
    f, _ = _features(jaipur_week)
    assert list(f.columns) == FEATURES
    assert "latitude" not in " ".join(f.columns).lower()   # model must not memorise location
    assert f.loc[f["solar_elevation"] > 5, "poa_kwm2"].max() < 1.5


def test_quantile_model_monotone_capped_and_roundtrips(jaipur_week, tmp_path):
    f, ideal = _features(jaipur_week, dc_ac=1.25)
    rng = np.random.default_rng(0)
    y = (ideal["ac"] * 0.9 * rng.uniform(0.7, 1.05, len(ideal))).clip(lower=0)
    params = dict(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.5, 0.9]),
                  n_estimators=40, max_depth=3, tree_method="hist")
    m = fit(f, y, "unit", params=params)
    q = m.predict(f)
    assert (q["p10"] <= q["p50"] + 1e-9).all() and (q["p50"] <= q["p90"] + 1e-9).all()
    assert (q.max(axis=1) <= 1 / 1.25 + 1e-6).all()
    assert (q.loc[f["solar_elevation"] <= 0] == 0).all().all()

    m.save(tmp_path)
    m2 = QuantileModel.load("unit", tmp_path)
    pd.testing.assert_frame_equal(m2.predict(f), q)


def test_score_metrics():
    y = pd.Series([0.0, 0.5, 1.0])
    s = score(y, y + 0.1, y - 0.2, y + 0.2)
    assert s["nrmse_pct"] == pytest.approx(10.0)
    assert s["coverage_p10_p90_pct"] == 100.0
