"""The additive model must be exactly the boosted model it was folded from, and the
reference decision logic must agree with it."""
import numpy as np
import pandas as pd
import pytest

from forewarn.gam import AdditiveModel, check_monotone, fit_additive, platt
from forewarn.registry import BY_NAME, model_features


@pytest.fixture(scope="module")
def fitted(snapshots):
    snap, lab, _ = snapshots
    df = pd.concat([snap, lab.drop(columns=["customer_id", "snapshot_month"])], axis=1)
    tr = df[(df["split"] == "train") & (df["weight"] > 0)]
    va = df[(df["split"] == "valid") & (df["weight"] > 0)]
    feats = model_features()
    model, booster = fit_additive(tr, tr["y_dlq"], tr["weight"], va, va["y_dlq"], va["weight"], feats, "delinquency", rounds=300, min_leaf=100)
    return model, booster, feats, df


def test_folded_tables_reproduce_the_boosted_model_exactly(fitted):
    model, booster, feats, df = fitted
    X = df.sample(20_000, random_state=0)
    assert X[feats].isna().any().any()                     # the check must cover missing values
    want = booster.predict(X[feats], raw_score=True, num_iteration=booster.best_iteration)
    assert np.abs(model.raw(X) - want).max() < 1e-9


def test_every_constrained_term_is_monotone(fitted):
    model, _, _, _ = fitted
    assert check_monotone(model) == []
    assert any(BY_NAME[t.feature].monotone != 0 for t in model.terms)
    # the check has teeth: reverse one table and it must complain
    broken = AdditiveModel.from_json(model.to_json())
    t = next(t for t in broken.terms if BY_NAME[t.feature].monotone != 0 and len(t.values) > 2 and np.ptp(t.values) > 0)
    t.values = t.values[::-1].copy()
    assert check_monotone(broken) != []


def test_terms_are_centred_on_the_training_population(fitted):
    model, _, _, df = fitted
    tr = df[(df["split"] == "train") & (df["weight"] > 0)]
    mean = np.average(model.contributions(tr), axis=0, weights=tr["weight"])
    assert np.abs(mean).max() < 1e-9                       # so a contribution reads as "versus the average account"


def test_json_round_trip_changes_nothing(fitted):
    model, _, _, df = fitted
    X = df.sample(5000, random_state=1)
    again = AdditiveModel.from_json(model.to_json())
    assert np.array_equal(model.raw(X), again.raw(X))


def test_platt_calibration_recovers_a_known_shift():
    rng = np.random.default_rng(0)
    raw = rng.normal(-3, 1.5, 300_000)
    y = (rng.random(300_000) < 1 / (1 + np.exp(-(0.4 + 0.8 * raw)))).astype(int)
    a, b = platt(raw, y)
    assert abs(a - 0.4) < 0.05 and abs(b - 0.8) < 0.03
