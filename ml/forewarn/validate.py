"""Independent-style validation evidence, computed on the out-of-time window.

Produces artifacts/metrics/validation.json; report.py renders it.  Sections map
to what a model-risk function asks of any model, vendor or in-house:

  outcomes analysis    discrimination, capture, calibration, stability by month
  benchmarking         champion, card-only and unconstrained comparators
  conceptual soundness term shapes, declared directions, importance
  lead time            how long before the first missed payment the alert fires
  separate target      what a delinquency score misses on bankruptcy filers
  reason codes         what the adverse-action reasons look like in practice
  fair lending         outcome and accuracy differences by protected group,
                       proxy screen, less-discriminatory-alternative search
  drift                population stability of features and scores
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import metrics as M
from .features import make_panel
from .gam import AdditiveModel, fit_additive
from .registry import BY_NAME, model_features
from .score import decide
from .store import Store
from .train import ALERT_SHARE, TARGETS, load_dataset, weighted_quantile

LEAD_BUCKETS = [15, 45, 75, 105, 135, 165, 195]      # days before the first missed payment
LOOKBACK = 8                                          # months of snapshots examined before the miss
CLEAN_MONTHS = 12                                     # "first" means no missed payment in this many prior months
HORIZON_MONTHS = 6


def _discrimination(y, s, w, groups, loss=None, loss_ok=None, share=0.05, n_boot=200) -> dict:
    out = {"auc": M.auc(y, s, w), "ks": M.ks(y, s, w)}
    out["gini"] = 2 * out["auc"] - 1
    lo, hi = M.cluster_bootstrap(lambda m: M.auc(y, s, w * m), groups, n_boot)
    out["auc_ci95"] = [lo, hi]
    for top in sorted({share, 0.10}):
        c = M.capture(y, s, w, top)
        if loss is not None:
            # dollars are measured only on snapshots with a full 12 months of outcome
            cl = M.capture(y[loss_ok], s[loss_ok], w[loss_ok], top, loss[loss_ok])
            c["amount_capture"], c["amount_rows"] = cl["amount_capture"], int(loss_ok.sum())
        out[f"top_{int(top * 100)}pct"] = c
    return out


def _wmedian(x, w):
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    return float(x[order][np.searchsorted(cw, 0.5)])


def _customer_weights(df: pd.DataFrame, ids: np.ndarray) -> np.ndarray:
    """One weight per customer: pilot-treated customers drop out, controls count 1/propensity."""
    per = df.drop_duplicates("customer_id").set_index("customer_id")[["arm", "propensity"]].reindex(ids)
    arm = per["arm"].fillna("not_enrolled").to_numpy()
    prop = per["propensity"].fillna(1.0).to_numpy()
    return np.where(arm == "not_enrolled", 1.0, np.where(arm == "none", 1.0 / prop, 0.0))


def _lead_time(df, scores, statements, truth, thresholds: dict, test_start: int) -> dict:
    panel = make_panel(statements)
    N, T = len(panel.customer_ids), panel.n_months
    si = np.searchsorted(panel.customer_ids, statements["customer_id"].to_numpy())
    sm = panel.month_index(statements["statement_date"])
    cpd = np.zeros((N, T), np.int8)
    cpd[si, sm] = statements["cycles_past_due"].to_numpy()

    ri = np.searchsorted(panel.customer_ids, scores["customer_id"].to_numpy())
    rt = scores["snapshot_month"].to_numpy().astype(np.int64)
    grids = {}
    for name, (col, thr) in thresholds.items():
        g = np.zeros((N, T), bool)
        g[ri, rt] = scores[col].to_numpy() >= thr
        grids[name] = g
    has_row = np.zeros((N, T), bool)
    has_row[ri, rt] = True

    # A missed payment that starts a spell reaching 60+ days past due.
    prev = np.concatenate([np.zeros((N, 1), np.int8), cpd[:, :-1]], axis=1)
    nxt = np.concatenate([cpd[:, 1:], np.zeros((N, 1), np.int8)], axis=1)
    start = (cpd == 1) & (prev == 0) & (nxt >= 2)
    lo, hi = test_start + LOOKBACK, T - 2
    start[:, :lo] = False
    start[:, hi + 1:] = False
    ci, fm = np.nonzero(start)
    keep = np.r_[True, ci[1:] != ci[:-1]]                # first qualifying spell per customer
    ci, fm = ci[keep], fm[keep]
    # Had the account missed any payment in the 12 months before?  If it had, an alert
    # "before the miss" may only be echoing that earlier miss, so those events are
    # reported separately and the headline uses accounts with a clean year behind them.
    missed = np.concatenate([np.zeros((N, 1)), np.cumsum(cpd >= 1, axis=1)], axis=1)
    recent_miss = (missed[ci, fm] - missed[ci, np.maximum(fm - CLEAN_MONTHS, 0)]) > 0

    w = _customer_weights(df, panel.customer_ids)[ci]
    ok = w > 0
    ci, fm, w, recent_miss = ci[ok], fm[ok], w[ok], recent_miss[ok]

    jl = truth.set_index("customer_id")["job_loss_month"].reindex(panel.customer_ids).to_numpy()[ci]
    cause_job = (jl > -900) & (jl <= fm) & (fm - jl <= 14)

    def summarize(g, sel):
        lead = np.full(len(ci), -1.0)
        for k in range(LOOKBACK, 0, -1):                 # earliest month first
            t = fm - k
            hit = g[ci, t] & (lead < 0)
            lead[hit] = (k - 1) * 30 + 15
        lead, ww, job = lead[sel], w[sel], cause_job[sel]
        alerted = lead > 0
        res = {
            "events": int(sel.sum()),
            "alerted_before_miss": float(np.average(alerted, weights=ww)),
            "median_lead_days_when_alerted": _wmedian(lead[alerted], ww[alerted]) if alerted.any() else None,
            "share_with_lead_at_least": {str(b): float(np.average(lead >= b, weights=ww)) for b in LEAD_BUCKETS},
        }
        for label, m in (("job_loss", job), ("other_causes", ~job)):
            if m.sum() > 20:
                res[f"lead_75d_plus_{label}"] = float(np.average(lead[m] >= 75, weights=ww[m]))
                res[f"n_{label}"] = int(m.sum())
        return res

    out = {"events": int((~recent_miss).sum()), "events_with_recent_miss": int(recent_miss.sum()),
           "clean_months": CLEAN_MONTHS, "first_miss_months": [int(lo), int(hi)],
           "lookback_months": LOOKBACK, "buckets_days": LEAD_BUCKETS, "models": {}, "recent_miss": {}}
    for name, g in grids.items():
        out["models"][name] = summarize(g, ~recent_miss)
        out["recent_miss"][name] = summarize(g, recent_miss)
    return out


def _filers(df, scores, statements, outcomes, budgets: dict, test_start: int) -> dict:
    """Per filer, not per account-month: who was never 60+ days past due before filing, and
    which score flagged them in the six months before they filed."""
    panel = make_panel(statements)
    N, T = len(panel.customer_ids), panel.n_months
    si = np.searchsorted(panel.customer_ids, statements["customer_id"].to_numpy())
    sm = panel.month_index(statements["statement_date"])
    cpd = np.zeros((N, T), np.int8)
    cpd[si, sm] = statements["cycles_past_due"].to_numpy()
    has = outcomes["bankruptcy_filing_date"].notna().to_numpy()
    fi = np.searchsorted(panel.customer_ids, outcomes["customer_id"].to_numpy()[has])
    fm = panel.month_index(outcomes["bankruptcy_filing_date"][has])
    sel = fm >= test_start + HORIZON_MONTHS                 # six out-of-time snapshots precede the filing
    fi, fm = fi[sel], fm[sel]
    w = _customer_weights(df, panel.customer_ids)[fi]
    ok = w > 0
    fi, fm, w = fi[ok], fm[ok], w[ok]
    serious = np.concatenate([np.zeros((N, 1)), np.cumsum(cpd >= 2, axis=1)], axis=1)
    clean = (serious[fi, fm] - serious[fi, np.maximum(fm - CLEAN_MONTHS, 0)]) == 0

    ri = np.searchsorted(panel.customer_ids, scores["customer_id"].to_numpy())
    rt = scores["snapshot_month"].to_numpy().astype(np.int64)
    out = {"filers": int(len(fi)), "clean_filers": int(clean.sum()), "filing_months": [int(test_start + HORIZON_MONTHS), int(T - 1)],
           "share_never_60dpd_in_prior_12m": float(np.average(clean, weights=w)), "flagged_before_filing": {}}
    for label, rules in budgets.items():
        g = np.zeros((N, T), bool)
        for col, thr in rules:
            g[ri, rt] |= scores[col].to_numpy() >= thr
        hit = np.zeros(len(fi), bool)
        for k in range(1, HORIZON_MONTHS + 1):
            hit |= g[fi, fm - k]
        out["flagged_before_filing"][label] = {
            "clean_filers": float(np.average(hit[clean], weights=w[clean])),
            "all_filers": float(np.average(hit, weights=w)),
        }
    return out


def _fairness(te: pd.DataFrame, prot: pd.DataFrame, feats: list[str]) -> dict:
    d = te.merge(prot, on="customer_id", how="left")
    w = d["weight"].to_numpy()
    res: dict = {"attributes": {}, "note": "Protected attributes are synthetic. group_a is correlated with income by construction; group_b is independent of everything and serves as a negative control."}
    for attr in ("group_a", "group_b"):
        g = d[attr].to_numpy().astype(bool)
        block = {"share_protected": float(np.average(g, weights=w)), "models": {}}
        for name, y in TARGETS.items():
            z, alert, yy = d[f"z_{name}"].to_numpy(), d[f"alert_{name}"].to_numpy(), d[y].to_numpy()
            p = d[f"p_{name}"].to_numpy()
            rp, rr = np.average(alert[g], weights=w[g]), np.average(alert[~g], weights=w[~g])
            sd = np.sqrt(np.average((z - np.average(z, weights=w)) ** 2, weights=w))
            neg_p, neg_r = g & (yy == 0), ~g & (yy == 0)
            pos_p, pos_r = g & (yy == 1), ~g & (yy == 1)
            block["models"][name] = {
                "alert_rate_protected": float(rp),
                "alert_rate_reference": float(rr),
                "adverse_rate_ratio": float(rp / rr),
                "favorable_outcome_ratio_AIR": float((1 - rp) / (1 - rr)),
                "standardized_mean_difference_logit": float((np.average(z[g], weights=w[g]) - np.average(z[~g], weights=w[~g])) / sd),
                "bad_rate_protected": float(np.average(yy[g], weights=w[g])),
                "bad_rate_reference": float(np.average(yy[~g], weights=w[~g])),
                "auc_protected": M.auc(yy[g], z[g], w[g]),
                "auc_reference": M.auc(yy[~g], z[~g], w[~g]),
                "observed_over_predicted_protected": float(np.average(yy[g], weights=w[g]) / np.average(p[g], weights=w[g])),
                "observed_over_predicted_reference": float(np.average(yy[~g], weights=w[~g]) / np.average(p[~g], weights=w[~g])),
                "false_positive_rate_protected": float(np.average(alert[neg_p], weights=w[neg_p])),
                "false_positive_rate_reference": float(np.average(alert[neg_r], weights=w[neg_r])),
                "true_positive_rate_protected": float(np.average(alert[pos_p], weights=w[pos_p])),
                "true_positive_rate_reference": float(np.average(alert[pos_r], weights=w[pos_r])),
            }
        proxy = []
        for f in feats:
            x = d[f].to_numpy(dtype=np.float64)
            okx = ~np.isnan(x)
            a = M.auc(g[okx], x[okx], w[okx]) if okx.sum() > 100 else float("nan")
            proxy.append({"feature": f, "auc_predicting_group": a, "strength": abs(a - 0.5),
                          "missing_rate_protected": float(np.average(~okx[g], weights=w[g])),
                          "missing_rate_reference": float(np.average(~okx[~g], weights=w[~g]))})
        proxy.sort(key=lambda r: -r["strength"])
        block["proxy_screen_top"] = proxy[:6]
        block["score_auc_predicting_group"] = {n: M.auc(g, d[f"z_{n}"].to_numpy(), w) for n in TARGETS}
        res["attributes"][attr] = block
    return res


def _lda_search(df: pd.DataFrame, prot: pd.DataFrame, feats: list[str], drop_candidates: list[str], sample: float, seed: int) -> list[dict]:
    """Refit the delinquency model without each candidate feature and compare accuracy and disparity."""
    fit = df["weight"] > 0
    tr = df[(df["split"] == "train") & fit].sample(frac=sample, random_state=seed)
    va = df[(df["split"] == "valid") & fit]
    te = df[(df["split"] == "test") & fit].merge(prot, on="customer_id", how="left")
    g, w, y = te["group_a"].to_numpy().astype(bool), te["weight"].to_numpy(), te["y_dlq"].to_numpy()
    rows = []
    for drop in [None] + drop_candidates:
        fs = [f for f in feats if f != drop]
        model, _ = fit_additive(tr, tr["y_dlq"], tr["weight"], va, va["y_dlq"], va["weight"], fs, "lda", seed=seed)
        s_va, s_te = model.raw(va), model.raw(te)
        thr = weighted_quantile(s_va, va["weight"].to_numpy(), 1 - ALERT_SHARE["delinquency"])
        alert = s_te >= thr
        rp, rr = np.average(alert[g], weights=w[g]), np.average(alert[~g], weights=w[~g])
        rows.append({"dropped": drop or "(none)", "auc": M.auc(y, s_te, w), "adverse_rate_ratio": float(rp / rr),
                     "capture_top5": M.capture(y, s_te, w, 0.05)["event_capture"]})
    return rows


def validate(store: Store, n_boot: int = 200, lda_sample: float = 0.25, seed: int = 0, log=print) -> dict:
    df = load_dataset(store)
    scores = pd.read_parquet(store.work / "scores.parquet")
    bundle = json.loads((store.artifacts / "bundle" / "bundle.json").read_text())
    feats = model_features()
    models = {n: AdditiveModel.from_json(bundle["models"][n]) for n in TARGETS}
    n_months = int(df["snapshot_month"].max()) + 1
    test_start = int(df.loc[df["split"] == "test", "snapshot_month"].min())

    thr_p = {n: bundle["alerts"][n]["threshold_probability"] for n in TARGETS}
    for n in TARGETS:
        scores[f"alert_{n}"] = scores[f"p_{n}"] >= thr_p[n]
        a, b = bundle["models"][n]["calibration"]["a"], bundle["models"][n]["calibration"]["b"]
        scores[f"z_{n}"] = a + b * scores[f"raw_{n}"]
    vmask = (scores["split"] == "valid") & (scores["weight"] > 0)
    champ_thr = {n: weighted_quantile(scores.loc[vmask, f"champion_{n}"].to_numpy(), scores.loc[vmask, "weight"].to_numpy(), 1 - ALERT_SHARE[n]) for n in TARGETS}

    tmask = ((scores["split"] == "test") & (scores["weight"] > 0)).to_numpy()
    te, te_f = scores[tmask].reset_index(drop=True), df[tmask].reset_index(drop=True)
    w, groups = te["weight"].to_numpy(), te["customer_id"].to_numpy()
    out: dict = {
        "data_source": bundle["data_source"],
        "bundle_sha256": (store.artifacts / "bundle" / "bundle.sha256").read_text().strip(),
        "windows": bundle["training"]["snapshot_months"],
        "rows": {s: int(((scores["split"] == s) & (scores["weight"] > 0)).sum()) for s in ("train", "valid", "test")},
        "customers_test": int(te["customer_id"].nunique()),
        "base_rates": {s: {n: float(np.average(scores.loc[(scores["split"] == s) & (scores["weight"] > 0), y],
                                               weights=scores.loc[(scores["split"] == s) & (scores["weight"] > 0), "weight"]))
                           for n, y in TARGETS.items()} for s in ("train", "valid", "test")},
        "models": {},
    }

    # ------------------------------------------------------------------ outcomes analysis and benchmarking
    for name, y in TARGETS.items():
        yy = te[y].to_numpy()
        loss, loss_ok = te["loss_12m"].to_numpy(), te["observed_12m"].to_numpy()
        block = {"alert_threshold": thr_p[name], "comparators": {}}
        for label, col in (("forewarn", f"p_{name}"), ("champion", f"champion_{name}"), ("card_only", f"card_only_{name}"),
                           ("benchmark_unconstrained", f"benchmark_{name}"), ("with_restricted_feature", f"restricted_{name}")):
            block["comparators"][label] = _discrimination(yy, te[col].to_numpy(), w, groups, loss, loss_ok, ALERT_SHARE[name], n_boot if label in ("forewarn", "champion") else 50)
        p = te[f"p_{name}"].to_numpy()
        block["calibration_test"] = M.calibration_table(yy, p, w).to_dict("records")
        va = scores[vmask]
        block["calibration_valid"] = M.calibration_table(va[y].to_numpy(), va[f"p_{name}"].to_numpy(), va["weight"].to_numpy()).to_dict("records")
        block["observed_over_predicted_test"] = float(np.average(yy, weights=w) / np.average(p, weights=w))
        block["brier_test"] = M.brier(yy, p, w)
        by_month = []
        for m, idx in te.groupby("snapshot_month").indices.items():
            by_month.append({
                "month": int(m), "n": int(len(idx)),
                "auc": M.auc(yy[idx], p[idx], w[idx]),
                "auc_champion": M.auc(yy[idx], te[f"champion_{name}"].to_numpy()[idx], w[idx]),
                "auc_benchmark": M.auc(yy[idx], te[f"benchmark_{name}"].to_numpy()[idx], w[idx]),
                "observed": float(np.average(yy[idx], weights=w[idx])),
                "predicted": float(np.average(p[idx], weights=w[idx])),
                "alert_rate": float(np.average(te[f"alert_{name}"].to_numpy()[idx], weights=w[idx])),
            })
        block["by_month"] = by_month
        has_dep = te_f["buffer_days"].notna().to_numpy()
        block["segments"] = {
            "with_deposit_history": {"share": float(np.average(has_dep, weights=w)), "auc": M.auc(yy[has_dep], p[has_dep], w[has_dep]),
                                     "auc_champion": M.auc(yy[has_dep], te[f"champion_{name}"].to_numpy()[has_dep], w[has_dep])},
            "card_only_customers": {"share": float(np.average(~has_dep, weights=w)), "auc": M.auc(yy[~has_dep], p[~has_dep], w[~has_dep]),
                                    "auc_champion": M.auc(yy[~has_dep], te[f"champion_{name}"].to_numpy()[~has_dep], w[~has_dep])},
        }
        # conceptual soundness
        mdl = models[name]
        C = mdl.contributions(te_f)
        terms = []
        for j, t in enumerate(mdl.terms):
            f = BY_NAME[t.feature]
            terms.append({
                "feature": t.feature, "group": f.group, "declared_direction": f.monotone,
                "importance": float(np.average(np.abs(C[:, j]), weights=w)),
                "range_logit": float(max(t.values.max(), t.missing) - min(t.values.min(), t.missing)),
                "bins": int(len(t.values)),
                "missing_contribution": float(t.missing),
                "monotone_ok": bool((np.diff(t.values) * f.monotone >= -1e-12).all()) if f.monotone else None,
                "shape": {"thresholds": t.thresholds.tolist(), "values": t.values.tolist()},
                "p01": float(np.nanpercentile(te_f[t.feature], 1)), "p99": float(np.nanpercentile(te_f[t.feature], 99)),
            })
        terms.sort(key=lambda r: -r["importance"])
        block["terms"] = terms
        block["features_unused"] = [f for f in feats if f not in mdl.features]
        out["models"][name] = block
        c = block["comparators"]
        log(f"{name}: AUC {c['forewarn']['auc']:.4f} [{c['forewarn']['auc_ci95'][0]:.4f}, {c['forewarn']['auc_ci95'][1]:.4f}]  champion {c['champion']['auc']:.4f}  benchmark {c['benchmark_unconstrained']['auc']:.4f}  O/E {block['observed_over_predicted_test']:.2f}")

    # ------------------------------------------------------------------ bankruptcy as a separate target
    statements = store.read_raw("card_statements")
    outcomes = store.read_raw("outcomes")
    y_bk, y_dlq = te["y_bk"].to_numpy(), te["y_dlq"].to_numpy()
    clean_rows = (y_bk == 1) & (y_dlq == 0)
    mask = clean_rows | (y_bk == 0)
    sep: dict = {"auc_account_months": {}}
    budgets: dict = {}
    for label, col in (("bankruptcy_score", "p_bankruptcy"), ("delinquency_score", "p_delinquency"), ("champion_delinquency", "champion_delinquency")):
        s = te[col].to_numpy()
        # pre-filing account-months with no 60+ DPD in their window, against all non-filer account-months
        sep["auc_account_months"][label] = M.auc(clean_rows[mask], s[mask], w[mask])
        for top in (0.02, 0.05):
            budgets[f"{label}_top{int(top * 100)}pct"] = [(col, weighted_quantile(s, w, 1 - top))]
    budgets["shipped_delinquency_alert"] = [("p_delinquency", thr_p["delinquency"])]
    budgets["shipped_both_alerts"] = [("p_delinquency", thr_p["delinquency"]), ("p_bankruptcy", thr_p["bankruptcy"])]
    sep.update(_filers(df, scores, statements, outcomes, budgets, test_start))
    both = te["alert_delinquency"].to_numpy() | te["alert_bankruptcy"].to_numpy()
    sep["combined_alert_rate"] = float(np.average(both, weights=w))
    out["separate_target"] = sep
    fb = sep["flagged_before_filing"]
    log(f"filers: {sep['filers']} ({sep['share_never_60dpd_in_prior_12m']:.1%} never 60+ DPD in the prior 12 months); of those, flagged before filing at a 5% budget: "
        f"bankruptcy score {fb['bankruptcy_score_top5pct']['clean_filers']:.1%}, delinquency score {fb['delinquency_score_top5pct']['clean_filers']:.1%}")

    # ------------------------------------------------------------------ alert burden
    bad = (y_dlq == 1) | (y_bk == 1)
    out["alert_burden"] = {
        "alerts_per_1000_accounts_per_month": float(1000 * np.average(both, weights=w)),
        "precision_bad_within_6m": float((w * both * bad).sum() / (w * both).sum()),
        "precision_missed_payment_or_bankruptcy_within_6m": float((w * both * (bad | (te["y_first_miss"].to_numpy() == 1))).sum() / (w * both).sum()),
        "recall_bad_within_6m": float((w * both * bad).sum() / (w * bad).sum()),
        "base_rate_bad": float(np.average(bad, weights=w)),
        "share_of_12m_loss_dollars_in_alerts": float((w * both * te["loss_12m"].to_numpy() * te["observed_12m"].to_numpy()).sum()
                                                     / (w * te["loss_12m"].to_numpy() * te["observed_12m"].to_numpy()).sum()),
        "loss_dollar_snapshot_months": [int(te.loc[te["observed_12m"], "snapshot_month"].min()), int(te.loc[te["observed_12m"], "snapshot_month"].max())],
    }

    # ------------------------------------------------------------------ lead time
    truth = store.read_raw("sim_truth")
    out["lead_time"] = _lead_time(
        df, scores, statements, truth,
        {"forewarn_delinquency": ("p_delinquency", thr_p["delinquency"]),
         "champion_delinquency": ("champion_delinquency", champ_thr["delinquency"])},
        test_start,
    )
    lt = out["lead_time"]["models"]["forewarn_delinquency"]
    log(f"lead time ({out['lead_time']['events']} first misses after a clean year): alerted before the miss {lt['alerted_before_miss']:.1%}; >=75 days {lt['share_with_lead_at_least']['75']:.1%}; >=195 days {lt['share_with_lead_at_least']['195']:.1%}")

    # ------------------------------------------------------------------ decisions, reasons
    names = [f["name"] for f in bundle["features"]]
    idx_alert = np.flatnonzero(both)
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx_alert, min(20_000, len(idx_alert)), replace=False)
    d = decide(bundle, te_f.iloc[pick][names].reset_index(drop=True))
    rs = {}
    for n in TARGETS:
        al = d[n]["alert"]
        lists = [r for r, a_ in zip(d[n]["reasons"], al) if a_]
        first = pd.Series([r[0]["text"] for r in lists if r]).value_counts(normalize=True).head(8)
        anyr = pd.Series([x["text"] for r in lists for x in r]).value_counts()
        rs[n] = {
            "alerted_sampled": int(al.sum()),
            "mean_reasons": float(np.mean([len(r) for r in lists])) if lists else 0.0,
            "share_with_no_reason": float(np.mean([len(r) == 0 for r in lists])) if lists else 0.0,
            "top_first_reason": {k: float(v) for k, v in first.items()},
            "share_of_alerts_citing": {k: float(v / max(len(lists), 1)) for k, v in anyr.head(10).items()},
        }
    out["reason_codes"] = rs
    wp = w[pick]
    wshare = lambda v: (pd.Series(wp).groupby(pd.Series(v)).sum() / wp.sum()).sort_values(ascending=False).round(4).to_dict()  # noqa: E731
    out["actions_on_alerted"] = {
        "mix": wshare(d["action"]["recommended"]),
        "notes": wshare(d["action"]["note"]),
        "adverse_action_notice_share": float(np.average(d["action"]["adverse_action_notice"], weights=wp)),
    }

    # ------------------------------------------------------------------ fair lending
    prot = store.read_raw("protected_attributes")
    te_fair = pd.concat([te_f[["customer_id"] + feats], te[["weight", "y_dlq", "y_bk", "p_delinquency", "p_bankruptcy",
                                                            "z_delinquency", "z_bankruptcy", "alert_delinquency", "alert_bankruptcy"]]], axis=1)
    out["fair_lending"] = _fairness(te_fair, prot, feats)
    cands = [r["feature"] for r in out["fair_lending"]["attributes"]["group_a"]["proxy_screen_top"][:3]]
    out["fair_lending"]["lda_search"] = {"training_sample": lda_sample, "model": "delinquency", "rows": _lda_search(df, prot, feats, cands, lda_sample, seed)}
    fa = out["fair_lending"]["attributes"]["group_a"]["models"]["delinquency"]
    log(f"fair lending (group_a, delinquency): alert rate {fa['alert_rate_protected']:.3%} vs {fa['alert_rate_reference']:.3%}, ratio {fa['adverse_rate_ratio']:.2f}; O/E {fa['observed_over_predicted_protected']:.2f} vs {fa['observed_over_predicted_reference']:.2f}")

    # ------------------------------------------------------------------ drift
    drift = {"features": [], "scores": {}}
    for f in names:
        ref = bundle["drift_reference"]["features"][f]
        cur = M.bin_shares(te_f[f].to_numpy(dtype=np.float64), np.array(ref["edges"]), w)
        drift["features"].append({"feature": f, "psi_train_to_test": M.psi(np.array(ref["shares"]), cur)})
    drift["features"].sort(key=lambda r: -r["psi_train_to_test"])
    for n in TARGETS:
        ref = bundle["drift_reference"]["scores"][n]
        z = te[f"z_{n}"].to_numpy()
        per = []
        for m, idx in te.groupby("snapshot_month").indices.items():
            per.append({"month": int(m), "psi": M.psi(np.array(ref["shares"]), M.bin_shares(z[idx], np.array(ref["edges"]), w[idx]))})
        drift["scores"][n] = {"psi_train_to_test": M.psi(np.array(ref["shares"]), M.bin_shares(z, np.array(ref["edges"]), w)), "by_month": per}
    # one month at a time, as a monthly batch run would see it
    single = []
    for m, idx in te.groupby("snapshot_month").indices.items():
        psis = []
        for f in names:
            ref = bundle["drift_reference"]["features"][f]
            psis.append((M.psi(np.array(ref["shares"]), M.bin_shares(te_f[f].to_numpy(dtype=np.float64)[idx], np.array(ref["edges"]), w[idx])), f))
        psis.sort(reverse=True)
        single.append({"month": int(m), "calendar_month": int((int(m) + int(str(store.read_raw("sim_config")["start"].iloc[0]).split("-")[1]) - 1) % 12 + 1),
                       "max_feature_psi": psis[0][0], "max_feature": psis[0][1],
                       "features_at_watch": int(sum(p >= 0.10 for p, _ in psis)), "features_at_alert": int(sum(p >= 0.25 for p, _ in psis)),
                       "score_psi": {n: next(r["psi"] for r in drift["scores"][n]["by_month"] if r["month"] == int(m)) for n in TARGETS}})
    drift["single_month"] = single
    out["drift"] = drift

    out["policy"] = store.read_json("metrics/policy_evaluation.json")
    out["export"] = store.read_json("metrics/export.json")
    out["train_summary"] = store.read_json("metrics/train_summary.json")
    sim = store.read_raw("sim_config").iloc[0].to_dict()
    out["simulator"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in sim.items()}
    out["n_months"] = n_months
    store.write_json("metrics/validation.json", out)
    return out
