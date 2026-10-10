"""Additive monotone models.

The risk scores are gradient-boosted decision stumps (trees of depth 1) with
monotone constraints.  A sum of stumps on single features is a generalized
additive model, so after training every tree is folded into one lookup table per
feature:

    score = intercept + f_1(x_1) + f_2(x_2) + ... + f_k(x_k)

Consequences that matter for model risk management:
  * the whole model prints on a few pages and can be read by a validator;
  * each feature's contribution to a decision is exact, not a SHAP estimate, so
    adverse-action reasons are derived from the scoring arithmetic itself;
  * the scorer is a binary search per feature, which is why the Go service has
    no ML dependency and reproduces these numbers to the last bit.
The price is that feature interactions are not modelled; the validation report
measures what that costs against an unconstrained benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .registry import BY_NAME


@dataclass
class Term:
    feature: str
    thresholds: np.ndarray           # sorted; bin i covers (thresholds[i-1], thresholds[i]]
    values: np.ndarray               # len(thresholds) + 1
    missing: float                   # contribution when the feature is NaN

    def contrib(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        out = np.full(x.shape, self.missing)
        ok = ~np.isnan(x)
        out[ok] = self.values[np.searchsorted(self.thresholds, x[ok], side="left")]
        return out

    @property
    def best(self) -> float:
        """Lowest-risk contribution this feature can make (reference for reason codes)."""
        return float(min(self.values.min(), self.missing))


@dataclass
class AdditiveModel:
    name: str
    link: str                        # "logit" or "identity"
    intercept: float
    terms: list[Term]
    calibration: tuple[float, float] = (0.0, 1.0)     # p = sigmoid(a + b * raw)
    meta: dict = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return [t.feature for t in self.terms]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        if not self.terms:                       # a constant model is legitimate (heavily regularized effects)
            return np.zeros((len(X), 0))
        return np.column_stack([t.contrib(X[t.feature].to_numpy()) for t in self.terms])

    def raw(self, X: pd.DataFrame) -> np.ndarray:
        return self.intercept + self.contributions(X).sum(axis=1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        r = self.raw(X)
        if self.link == "identity":
            return r
        a, b = self.calibration
        return 1.0 / (1.0 + np.exp(-(a + b * r)))

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "link": self.link,
            "intercept": float(self.intercept),
            "calibration": {"a": float(self.calibration[0]), "b": float(self.calibration[1])},
            "terms": [
                {
                    "feature": t.feature,
                    "thresholds": [float(v) for v in t.thresholds],
                    "values": [float(v) for v in t.values],
                    "missing": float(t.missing),
                }
                for t in self.terms
            ],
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, d: dict) -> "AdditiveModel":
        return cls(
            d["name"], d["link"], d["intercept"],
            [Term(t["feature"], np.array(t["thresholds"]), np.array(t["values"]), t["missing"]) for t in d["terms"]],
            (d["calibration"]["a"], d["calibration"]["b"]),
            d.get("meta", {}),
        )


def monotone_vector(features: list[str]) -> list[int]:
    return [BY_NAME[f].monotone for f in features]


def collapse(booster: lgb.Booster, features: list[str], X_ref: pd.DataFrame, w_ref=None,
             name: str = "", link: str = "logit") -> AdditiveModel:
    """Fold a depth-1 ensemble into one table per feature, centred on X_ref."""
    dump = booster.dump_model()
    stumps: dict[int, list[tuple[float, float, float, bool]]] = {}
    intercept = 0.0
    for tree in dump["tree_info"]:
        node = tree["tree_structure"]
        if "leaf_value" in node:                      # tree with no split
            intercept += node["leaf_value"]
            continue
        l, r = node["left_child"], node["right_child"]
        if "leaf_value" not in l or "leaf_value" not in r:
            raise ValueError("model is not additive: found a tree deeper than one split")
        if node["decision_type"] != "<=":
            raise ValueError(f"unsupported split type {node['decision_type']}")
        mt = node["missing_type"]
        if mt == "NaN":
            nan_left = bool(node["default_left"])
        elif mt == "None":                            # LightGBM scores NaN as 0.0 here
            nan_left = 0.0 <= node["threshold"]
        else:
            raise ValueError(f"unsupported missing_type {mt}")
        stumps.setdefault(node["split_feature"], []).append(
            (float(node["threshold"]), float(l["leaf_value"]), float(r["leaf_value"]), nan_left)
        )

    terms: list[Term] = []
    for j, feat in enumerate(features):
        if j not in stumps:
            continue
        thr = np.array([s[0] for s in stumps[j]])
        left = np.array([s[1] for s in stumps[j]])
        right = np.array([s[2] for s in stumps[j]])
        nan_left = np.array([s[3] for s in stumps[j]])
        u = np.unique(thr)
        pos = np.searchsorted(u, thr)
        # bin i gets the left value of every stump whose threshold index >= i, else the right value
        vals = np.zeros(len(u) + 1)
        for i in range(len(u) + 1):
            vals[i] = left[pos >= i].sum() + right[pos < i].sum()
        missing = float(np.where(nan_left, left, right).sum())
        terms.append(Term(feat, u, vals, missing))

    model = AdditiveModel(name, link, intercept, terms)
    # centre every term so the intercept is the average score and contributions read as deviations
    w = np.ones(len(X_ref)) if w_ref is None else np.asarray(w_ref, dtype=np.float64)
    for t in model.terms:
        mu = float(np.average(t.contrib(X_ref[t.feature].to_numpy()), weights=w))
        t.values = t.values - mu
        t.missing = t.missing - mu
        model.intercept += mu
    return model


def fit_additive(
    X_tr: pd.DataFrame, y_tr, w_tr,
    X_va: pd.DataFrame | None, y_va, w_va,
    features: list[str],
    name: str,
    objective: str = "binary",
    rounds: int = 4000,
    learning_rate: float = 0.12,
    min_leaf: int = 400,
    max_bin: int = 63,
    reg_lambda: float = 5.0,
    monotone: bool = True,
    seed: int = 0,
) -> tuple[AdditiveModel, lgb.Booster]:
    params = dict(
        objective=objective,
        learning_rate=learning_rate,
        num_leaves=2,
        max_depth=1,
        min_data_in_leaf=min_leaf,
        max_bin=max_bin,
        lambda_l2=reg_lambda,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        num_threads=2,
        seed=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    if monotone:
        params["monotone_constraints"] = monotone_vector(features)
    dtr = lgb.Dataset(X_tr[features], y_tr, weight=w_tr, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if X_va is not None:
        valid_sets = [lgb.Dataset(X_va[features], y_va, weight=w_va, reference=dtr)]
        callbacks.append(lgb.early_stopping(100, verbose=False))
    booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    link = "logit" if objective == "binary" else "identity"
    model = collapse(booster, features, X_tr, w_tr, name=name, link=link)
    model.meta = {
        "trees": int(booster.best_iteration or booster.current_iteration()),
        "learning_rate": learning_rate,
        "min_leaf": min_leaf,
        "monotone": monotone,
    }
    return model, booster


def platt(raw: np.ndarray, y, w=None) -> tuple[float, float]:
    """Logistic recalibration p = sigmoid(a + b * raw), fitted by Newton steps."""
    y = np.asarray(y, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    a, b = 0.0, 1.0
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(a + b * raw)))
        g = np.array([np.sum(w * (p - y)), np.sum(w * (p - y) * raw)])
        s = w * p * (1 - p)
        H = np.array([[s.sum(), (s * raw).sum()], [(s * raw).sum(), (s * raw * raw).sum()]])
        step = np.linalg.solve(H + 1e-9 * np.eye(2), g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def check_monotone(model: AdditiveModel) -> list[str]:
    """Return a message for every term whose table violates its declared direction."""
    bad = []
    for t in model.terms:
        direction = BY_NAME[t.feature].monotone
        d = np.diff(t.values)
        if direction > 0 and (d < -1e-12).any():
            bad.append(f"{model.name}.{t.feature}: expected non-decreasing")
        if direction < 0 and (d > 1e-12).any():
            bad.append(f"{model.name}.{t.feature}: expected non-increasing")
    return bad
