"""Which action, not only which account.

A risk score ranks accounts.  It does not say what to do, and the same action
helps one customer and costs money on another: a hardship plan rescues someone
between jobs and gives away interest on someone who was never going to default.

This module learns, from a randomized pilot, how the 18-month value of an
account (revenue - credit loss - action cost) changes under each action, as a
function of the same features the scores use:

    value_a(x) = m(x) + g_a(x)

  m    one prognostic model fitted on every arm (what kind of account is this)
  g_a  one small model per arm fitted on the residuals of that arm only (what
       the action changes); heavily regularized so that it collapses to the
       arm's average effect unless the data insist otherwise

Recommendation: the eligible action with the largest g_a(x) - g_none(x),
provided the estimated gain clears a minimum; otherwise no action.

Evaluation is cross-fitted and off-policy: each pilot account is assigned an
action by models that never saw it, and the policy is valued with inverse
propensity weighting on the accounts whose randomly assigned arm happens to
match.  Because arms were randomized with known probability, that estimate is
unbiased, and it does not depend on the value models being right.

Two policies are valued:
  forewarn_policy   the value models applied to every pilot account
  shipped rule      what the scorer actually does: the same choice, but only
                    for accounts that are alerted by a risk score and inside
                    pilot support.  It is valued on pilot decisions made after
                    the risk models' training window, so the alerts are out of
                    sample for every arm.

Requirements on real data: a randomized (or at least well-understood)
assignment of actions.  Historical actions chosen by collectors cannot be used
this way, because the accounts they chose to treat differ from the ones they
did not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ARMS
from .gam import AdditiveModel, fit_additive
from .registry import value_features
from .store import Store
from .train import TARGETS, load_dataset

VALUE_PARAMS = dict(
    base=dict(rounds=400, learning_rate=0.05, min_leaf=300, reg_lambda=10.0),
    arm=dict(rounds=120, learning_rate=0.05, min_leaf=250, reg_lambda=50.0),
    min_gain=50.0,
)

# Business eligibility, applied before any model output is considered.
ELIGIBILITY = {
    "due_date_change": [{"feature": "payroll_gap_days", "op": "<=", "value": 0.0}],      # paid within the last cycle
    "hardship_plan": [{"feature": "statement_balance", "op": ">=", "value": 200.0}],
    "limit_decrease": [{"feature": "open_to_buy", "op": ">=", "value": 200.0}],
    "outreach_call": [],
}
# The pilot enrolled accounts that tripped the bank's legacy trigger.  Outside it
# the value models are extrapolating, and the scorer says so.
PILOT_SUPPORT = [
    [{"feature": "od_count_3m", "op": ">=", "value": 1.0}],
    [{"feature": "payroll_gap_days", "op": ">", "value": 0.0}],
    [{"feature": "utilization", "op": ">=", "value": 0.5}, {"feature": "min_pay_months_6m", "op": ">=", "value": 2.0}],
    [{"feature": "cash_adv_months_6m", "op": ">=", "value": 1.0}],
    [{"feature": "util_change_3m", "op": ">=", "value": 0.15}],
]


def _cmp(x: np.ndarray, op: str, v: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        r = {"<=": x <= v, ">=": x >= v, "<": x < v, ">": x > v}[op]
    return r & ~np.isnan(x)          # a missing value never satisfies a condition


def eligible(X: pd.DataFrame) -> np.ndarray:
    """(n, len(ARMS)) boolean; 'none' is always eligible."""
    out = np.ones((len(X), len(ARMS)), bool)
    for j, arm in enumerate(ARMS):
        for c in ELIGIBILITY.get(arm, []):
            out[:, j] &= _cmp(X[c["feature"]].to_numpy(), c["op"], c["value"])
    return out


def in_pilot_support(X: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(X), bool)
    for clause in PILOT_SUPPORT:
        m = np.ones(len(X), bool)
        for c in clause:
            m &= _cmp(X[c["feature"]].to_numpy(), c["op"], c["value"])
        out |= m
    return out


def fit_value_models(D: pd.DataFrame, feats: list[str], params=VALUE_PARAMS, seed: int = 0):
    y = D["value_h"].to_numpy()
    base, _ = fit_additive(D, y, None, None, None, None, feats, "value_base", objective="regression",
                           monotone=False, seed=seed, **params["base"])
    resid = y - base.raw(D)
    arms: dict[str, AdditiveModel] = {}
    for a in ARMS:
        m = (D["arm"] == a).to_numpy()
        model, _ = fit_additive(D[m], resid[m], None, None, None, None, feats, f"value_effect_{a}",
                                objective="regression", monotone=False, seed=seed, **params["arm"])
        arms[a] = model
    return base, arms


def gains(arms: dict[str, AdditiveModel], X: pd.DataFrame) -> np.ndarray:
    """Estimated value of each action relative to no action, (n, len(ARMS))."""
    g = np.column_stack([arms[a].raw(X) for a in ARMS])
    return g - g[:, [0]]


def choose(gain: np.ndarray, elig: np.ndarray, min_gain: float) -> np.ndarray:
    g = np.where(elig, gain, -np.inf)
    g[:, 0] = 0.0
    best = g.argmax(axis=1)
    return np.where(g[np.arange(len(g)), best] >= min_gain, best, 0)


def expert_rules(X: pd.DataFrame) -> np.ndarray:
    """A plausible hand-written playbook, for comparison."""
    e = eligible(X)
    c = lambda f, op, v: _cmp(X[f].to_numpy(), op, v)  # noqa: E731
    out = np.zeros(len(X), np.int64)
    out = np.where(c("min_pay_months_6m", ">=", 3), 4, out)
    out = np.where(c("late_fee_count_6m", ">=", 2) & e[:, 1], 1, out)
    out = np.where((c("payroll_gap_days", ">", 4) | c("payroll_amt_ratio_3m", "<", 0.6)) & e[:, 2], 2, out)
    out = np.where(c("cash_adv_months_6m", ">=", 1) & c("open_to_buy", ">=", 500), 3, out)
    return out


def ipw_terms(logged: np.ndarray, policy: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-row Horvitz-Thompson terms; their mean is the policy value."""
    return (logged == policy) / p * y


def evaluate(store: Store, folds: int = 5, n_boot: int = 1000, seed: int = 0, params=VALUE_PARAMS, log=print) -> dict:
    df = load_dataset(store)
    D = df[df["is_decision_row"] & df["observed_h"]].reset_index(drop=True)
    feats = value_features()
    logged = pd.Categorical(D["arm"], categories=ARMS).codes.astype(np.int64)
    y = D["value_h"].to_numpy()
    p = D["propensity"].to_numpy()                       # recorded by the pilot, never assumed
    elig = eligible(D)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, len(D))

    # cross-fitted policy: every account is assigned by models that did not see it
    gain = np.zeros((len(D), len(ARMS)))
    for k in range(folds):
        tr, te = fold != k, fold == k
        _, arms = fit_value_models(D[tr], feats, params, seed)
        gain[te] = gains(arms, D[te])
    model_policy = choose(gain, elig, params["min_gain"])

    policies = {"no_action": np.zeros(len(D), np.int64)}
    for j, a in enumerate(ARMS[1:], start=1):
        policies[f"always_{a}"] = np.where(elig[:, j], j, 0)
    policies["expert_rules"] = expert_rules(D)
    policies["forewarn_policy"] = model_policy

    terms = {k: ipw_terms(logged, v, y, p) for k, v in policies.items()}
    base = terms["no_action"]
    boot = rng.integers(0, len(D), (n_boot, len(D)))
    result: dict = {
        "n_decisions": int(len(D)),
        "horizon_months": 18,
        "value_definition": "revenue (interest, interchange, cash-advance fees; penalty fees excluded) minus credit loss minus action cost, per pilot account",
        "arms": {},
        "policies": {},
        "params": params,
    }
    for j, a in enumerate(ARMS):
        m = logged == j
        result["arms"][a] = {
            "n": int(m.sum()),
            "mean_value": float(y[m].mean()),
            "se": float(y[m].std() / np.sqrt(m.sum())),
            "loss": float(D.loc[m, "loss_h"].mean()),
            "revenue": float(D.loc[m, "revenue_h"].mean()),
            "bad_rate": float((D.loc[m, "loss_h"] > 0).mean()),
        }
    for k, tm in terms.items():
        d = tm - base
        bd = d[boot].mean(axis=1)
        mix = np.bincount(policies[k], minlength=len(ARMS)) / len(D)
        result["policies"][k] = {
            "value": float(tm.mean()),
            "gain_vs_no_action": float(d.mean()),
            "gain_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "p_gain_le_0": float((bd <= 0).mean()),
            "action_mix": dict(zip(ARMS, mix.round(4).tolist())),
        }
        log(f"  {k:28s} value {tm.mean():8.1f}  gain {d.mean():+7.1f}  95% CI [{np.percentile(bd, 2.5):+.1f}, {np.percentile(bd, 97.5):+.1f}]  mix {mix.round(2).tolist()}")

    # who the policy treats, and what it was worth on exactly those accounts
    by_action = {}
    for j, a in enumerate(ARMS[1:], start=1):
        sel = model_policy == j
        t_m, c_m = sel & (logged == j), sel & (logged == 0)
        if t_m.sum() < 2 or c_m.sum() < 2:
            continue
        diff = float(y[t_m].mean() - y[c_m].mean())
        se = float(np.sqrt(y[t_m].var(ddof=1) / t_m.sum() + y[c_m].var(ddof=1) / c_m.sum()))
        by_action[a] = {
            "share_of_pilot": float(sel.mean()),
            "n_treated_as_recommended": int(t_m.sum()),
            "n_control": int(c_m.sum()),
            "value_if_treated": float(y[t_m].mean()),
            "value_if_no_action": float(y[c_m].mean()),
            "difference": diff,
            "difference_se": se,
            "mean_predicted_gain": float(gain[sel, j].mean()),
        }
    result["forewarn_policy_by_action"] = by_action

    # The rule the scorer applies: act only on alerted accounts inside pilot support.
    # Alerts need risk scores, and the risk models trained on control-arm outcomes up to
    # the end of their training window, so only later pilot decisions are used here.
    train_end = int(df.loc[df["split"] == "train", "snapshot_month"].max())
    alert = np.zeros(len(D), bool)
    for name in TARGETS:
        mj = store.read_json(f"models/{name}.json")
        alert |= AdditiveModel.from_json(mj).predict(D) >= mj["meta"]["alert_threshold"]
    support = in_pilot_support(D)
    late = (D["snapshot_month"] > train_end).to_numpy()
    shipped = np.where(alert & support, model_policy, 0)
    d_s = (ipw_terms(logged, shipped, y, p) - base)[late]
    d_u = (terms["forewarn_policy"] - base)[late]
    bs = rng.integers(0, len(d_s), (n_boot, len(d_s)))
    result["shipped_rule"] = {
        "n_decisions": int(late.sum()),
        "decision_months": [int(D.loc[late, "snapshot_month"].min()), int(D.loc[late, "snapshot_month"].max())],
        "share_alerted": float(alert[late].mean()),
        "share_alerted_and_in_support": float((alert & support)[late].mean()),
        "share_treated": float((shipped[late] != 0).mean()),
        "action_mix": dict(zip(ARMS, (np.bincount(shipped[late], minlength=len(ARMS)) / late.sum()).round(4).tolist())),
        "gain_vs_no_action": float(d_s.mean()),
        "gain_ci95": [float(np.percentile(d_s[bs].mean(axis=1), 2.5)), float(np.percentile(d_s[bs].mean(axis=1), 97.5))],
        "p_gain_le_0": float((d_s[bs].mean(axis=1) <= 0).mean()),
        "ungated_policy_same_accounts": {
            "gain_vs_no_action": float(d_u.mean()),
            "gain_ci95": [float(np.percentile(d_u[bs].mean(axis=1), 2.5)), float(np.percentile(d_u[bs].mean(axis=1), 97.5))],
        },
    }
    sr = result["shipped_rule"]
    log(f"  shipped rule (alert + pilot support), {sr['n_decisions']:,} decisions in months {sr['decision_months'][0]}-{sr['decision_months'][1]}: "
        f"treats {sr['share_treated']:.1%}, gain {sr['gain_vs_no_action']:+.1f}  95% CI [{sr['gain_ci95'][0]:+.1f}, {sr['gain_ci95'][1]:+.1f}]")

    # final models on all pilot data go into the bundle
    base_model, arms = fit_value_models(D, feats, params, seed)
    store.write_json("models/value_base.json", base_model.to_json())
    for a, mdl in arms.items():
        store.write_json(f"models/value_effect_{a}.json", mdl.to_json())
    store.write_json("metrics/policy_evaluation.json", result)
    pd.DataFrame({"customer_id": D["customer_id"], "snapshot_month": D["snapshot_month"], "logged_arm": D["arm"],
                  "policy_arm": np.array(ARMS)[model_policy], "value_h": y,
                  **{f"gain_{a}": gain[:, j] for j, a in enumerate(ARMS)}}).to_parquet(store.work / "policy_crossfit.parquet", index=False)
    return result
