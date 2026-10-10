"""Reference implementation of the decision the Go scorer makes.

One function, `decide`, turns a feature frame into scores, alerts, reasons and
a recommended action using nothing but the model bundle.  The Go service is a
second implementation of exactly this file; `forewarn verify` replays the
parity cases exported from here and fails on any difference.

Arithmetic is written so both implementations are bit-identical where it
matters: contributions are summed in term order, and alerts compare logits
(no exponential) against a stored threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _contrib(term: dict, x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, term["missing"], dtype=np.float64)
    ok = ~np.isnan(x)
    out[ok] = np.asarray(term["values"])[np.searchsorted(np.asarray(term["thresholds"]), x[ok], side="left")]
    return out


def model_contributions(model: dict, X: pd.DataFrame) -> np.ndarray:
    if not model["terms"]:
        return np.zeros((len(X), 0))
    return np.column_stack([_contrib(t, X[t["feature"]].to_numpy(dtype=np.float64)) for t in model["terms"]])


def model_raw(model: dict, C: np.ndarray) -> np.ndarray:
    r = np.full(C.shape[0], model["intercept"], dtype=np.float64)
    for j in range(C.shape[1]):          # sequential, in term order
        r = r + C[:, j]
    return r


def _cmp(x, op, v):
    with np.errstate(invalid="ignore"):
        r = {"<=": x <= v, ">=": x >= v, "<": x < v, ">": x > v}[op]
    return r & ~np.isnan(x)


def reasons_for_row(model: dict, features: dict, c_row: np.ndarray, x_row: dict, max_reasons: int, min_points: float) -> list[dict]:
    """Principal reasons, one per reason group.

    Every term is centred on the training population, so a contribution is the
    number of log-odds points by which this account's value of the feature is
    riskier (positive) or safer (negative) than the average account's.  The
    reasons are the groups that add the most risk relative to that average,
    which is one of the methods the Regulation B commentary describes for
    scored decisions (comment 9(b)(2)-5).
    """
    groups: dict[str, dict] = {}
    for j, t in enumerate(model["terms"]):
        lost = c_row[j]
        f = features[t["feature"]]
        g = groups.setdefault(f["group"], {"points": 0.0, "top": -np.inf, "order": j, "feature": None})
        g["points"] = g["points"] + lost
        if lost > g["top"]:
            g["top"], g["feature"] = lost, t["feature"]
    ranked = sorted((g for g in groups.values() if g["points"] > min_points), key=lambda g: (-g["points"], g["order"]))
    out = []
    for g in ranked[:max_reasons]:
        f = features[g["feature"]]
        v = x_row.get(g["feature"])
        missing = v is None or (isinstance(v, float) and np.isnan(v))
        out.append({
            "code": f["reason_code"] + ("M" if missing else ""),
            "text": f["missing_text"] if missing else f["reason_text"],
            "feature": g["feature"],
            "points": float(g["points"]),
        })
    return out


def decide(bundle: dict, X: pd.DataFrame, with_reasons: bool = True) -> dict:
    """Vectorized decision.  Returns arrays plus, optionally, per-row reason lists."""
    feats = {f["name"]: f for f in bundle["features"]}
    n = len(X)
    for f in bundle["features"]:
        if f.get("required") and X[f["name"]].isna().any():
            raise ValueError(f"feature {f['name']!r} is required and has missing values")
    out: dict = {"n": n}
    alert_any = np.zeros(n, bool)
    margin = {}
    for name in ("delinquency", "bankruptcy"):
        m = bundle["models"][name]
        C = model_contributions(m, X)
        raw = model_raw(m, C)
        z = m["calibration"]["a"] + m["calibration"]["b"] * raw
        thr = bundle["alerts"][name]["threshold_logit"]
        alert = z >= thr
        out[name] = {"logit": z, "probability": 1.0 / (1.0 + np.exp(-z)), "alert": alert, "contributions": C}
        margin[name] = z - thr
        alert_any |= alert
    out["alert"] = alert_any
    out["driver"] = np.where(~alert_any, "", np.where(margin["delinquency"] >= margin["bankruptcy"], "delinquency", "bankruptcy"))

    pol = bundle["policy"]
    arms = pol["arms"]
    g = []
    for a in arms:
        m = bundle["models"]["value_effects"][a]
        g.append(model_raw(m, model_contributions(m, X)))
    g = np.column_stack(g)
    gain = g - g[:, [0]]
    elig = np.ones((n, len(arms)), bool)
    for j, a in enumerate(arms):
        for c in pol["eligibility"].get(a, []):
            elig[:, j] &= _cmp(X[c["feature"]].to_numpy(dtype=np.float64), c["op"], c["value"])
    support = np.zeros(n, bool)
    for clause in pol["pilot_support"]:
        ok = np.ones(n, bool)
        for c in clause:
            ok &= _cmp(X[c["feature"]].to_numpy(dtype=np.float64), c["op"], c["value"])
        support |= ok
    cand = np.where(elig, gain, -np.inf)
    cand[:, 0] = -np.inf
    best = cand.argmax(axis=1)                       # first maximum wins ties
    best_gain = cand[np.arange(n), best]
    clears = best_gain >= pol["min_gain"]
    rec = np.where(alert_any & support & clears, best, 0)
    note = np.where(~alert_any, "below_alert_threshold",
                    np.where(~support, "outside_pilot_support",
                             np.where(~clears, "no_action_clears_minimum_gain", "recommended")))
    out["action"] = {
        "recommended": np.asarray(arms)[rec],
        "note": note,
        "gain": gain,
        "eligible": elig,
        "in_pilot_support": support,
        "adverse_action_notice": np.isin(np.asarray(arms)[rec], pol["adverse_actions"]),
    }
    if with_reasons:
        rc = bundle["reasons"]
        cols = sorted({t["feature"] for name in ("delinquency", "bankruptcy") for t in bundle["models"][name]["terms"]})
        rows = X[cols].astype(object).where(X[cols].notna(), None).to_dict("records")
        for name in ("delinquency", "bankruptcy"):
            m, C = bundle["models"][name], out[name]["contributions"]
            out[name]["reasons"] = [
                reasons_for_row(m, feats, C[i], rows[i], rc["max_reasons"], rc["min_points"]) for i in range(n)
            ]
    return out


def row_view(bundle: dict, d: dict, i: int) -> dict:
    """The JSON document the scorer returns for row i."""
    arms = bundle["policy"]["arms"]
    scores = {}
    for name in ("delinquency", "bankruptcy"):
        scores[name] = {
            "probability": float(d[name]["probability"][i]),
            "logit": float(d[name]["logit"][i]),
            "alert": bool(d[name]["alert"][i]),
            "reasons": d[name].get("reasons", [[]] * d["n"])[i],
        }
    a = d["action"]
    return {
        "alert": bool(d["alert"][i]),
        "driver": str(d["driver"][i]),
        "scores": scores,
        "action": {
            "recommended": str(a["recommended"][i]),
            "note": str(a["note"][i]),
            "estimated_gain": {arm: float(a["gain"][i, j]) for j, arm in enumerate(arms)},
            "eligible": {arm: bool(a["eligible"][i, j]) for j, arm in enumerate(arms)},
            "in_pilot_support": bool(a["in_pilot_support"][i]),
            "adverse_action_notice": bool(a["adverse_action_notice"][i]),
        },
    }
