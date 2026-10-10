"""Train the two risk scores, their comparators, and score every snapshot.

  delinquency   P(60+ days past due within 6 months | current today)
  bankruptcy    P(bankruptcy filing within 6 months | current today)

Bankruptcy is its own target because a large share of filers never go
delinquent first: a delinquency model has nothing to learn from them (their
label is 0), so it ranks them low.  The validation report quantifies this.

Comparators trained on the same rows:
  champion    logistic regression on card-only behaviour fields: a stand-in for
              the behaviour score a bank already runs (it has no bureau data, so
              a real champion would be stronger than this one)
  card_only   the additive model restricted to card features: isolates what the
              deposit-transaction features add
  benchmark   unconstrained gradient boosting on the same features: an upper
              bound that shows what monotonicity and additivity cost
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import metrics as M
from .gam import AdditiveModel, check_monotone, fit_additive, platt
from .registry import BY_NAME, assert_permitted, model_features
from .store import Store

TARGETS = {"delinquency": "y_dlq", "bankruptcy": "y_bk"}
CHAMPION_FEATURES = [
    "utilization", "pay_ratio_3m", "late_fee_count_6m", "max_cpd_12m", "min_pay_months_6m", "cash_adv_months_6m",
]
ALERT_SHARE = {"delinquency": 0.05, "bankruptcy": 0.02}


def load_dataset(store: Store) -> pd.DataFrame:
    f = pd.read_parquet(store.work / "features.parquet")
    l = pd.read_parquet(store.work / "labels.parquet")
    assert (f["customer_id"].to_numpy() == l["customer_id"].to_numpy()).all()
    assert (f["snapshot_month"].to_numpy() == l["snapshot_month"].to_numpy()).all()
    return pd.concat([f, l.drop(columns=["customer_id", "snapshot_month"])], axis=1)


def weighted_quantile(x, w, q):
    order = np.argsort(x)
    cw = np.cumsum(w[order]) / w.sum()
    return float(x[order][np.searchsorted(cw, q)])


def _fit_champion(tr, y, feats):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=200))
    pipe.fit(tr[feats].to_numpy(dtype=np.float64), tr[y].to_numpy(), logisticregression__sample_weight=tr["weight"].to_numpy())
    return pipe


def _fit_benchmark(tr, va, y, feats, seed=0):
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, lambda_l2=5.0,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, num_threads=2, seed=seed, verbose=-1)
    dtr = lgb.Dataset(tr[feats], tr[y], weight=tr["weight"])
    dva = lgb.Dataset(va[feats], va[y], weight=va["weight"], reference=dtr)
    return lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


def train(store: Store, train_sample: float = 1.0, seed: int = 0, log=print) -> dict:
    df = load_dataset(store)
    feats = model_features()
    assert_permitted(feats)
    card_feats = [f for f in feats if BY_NAME[f].source == "card"]
    feats_r = model_features(include_restricted=True)

    fit = df["weight"] > 0
    tr = df[(df["split"] == "train") & fit]
    if train_sample < 1.0:
        tr = tr.sample(frac=train_sample, random_state=seed)
    va = df[(df["split"] == "valid") & fit]
    te = df[(df["split"] == "test") & fit]
    log(f"rows: train {len(tr):,}  valid {len(va):,}  test {len(te):,}")

    scores = df[["customer_id", "snapshot_month", "split", "weight", "y_dlq", "y_bk", "y_first_miss",
                 "loss_12m", "observed_6m", "observed_12m"]].copy()
    models: dict[str, AdditiveModel] = {}
    summary: dict = {"features": feats, "alert_share": ALERT_SHARE, "rows": {"train": len(tr), "valid": len(va), "test": len(te)}}

    for name, y in TARGETS.items():
        t0 = time.time()
        model, _ = fit_additive(tr, tr[y], tr["weight"], va, va[y], va["weight"], feats, name, seed=seed)
        model.calibration = platt(model.raw(va), va[y].to_numpy(), va["weight"].to_numpy())
        viol = check_monotone(model)
        if viol:
            raise AssertionError(f"monotonicity violated: {viol}")
        models[name] = model
        scores[f"raw_{name}"] = model.raw(df)
        scores[f"p_{name}"] = model.predict(df)
        log(f"{name}: additive model, {model.meta['trees']} stumps over {len(model.terms)} features, {time.time() - t0:.0f}s")

        champ = _fit_champion(tr, y, CHAMPION_FEATURES)
        scores[f"champion_{name}"] = champ.predict_proba(df[CHAMPION_FEATURES].to_numpy(dtype=np.float64))[:, 1]

        card_model, _ = fit_additive(tr, tr[y], tr["weight"], va, va[y], va["weight"], card_feats, f"{name}_card_only", seed=seed)
        scores[f"card_only_{name}"] = card_model.raw(df)

        bench = _fit_benchmark(tr, va, y, feats, seed)
        scores[f"benchmark_{name}"] = bench.predict(df[feats], num_iteration=bench.best_iteration)

        restricted, _ = fit_additive(tr, tr[y], tr["weight"], va, va[y], va["weight"], feats_r, f"{name}_with_restricted", seed=seed)
        scores[f"restricted_{name}"] = restricted.raw(df)

        vmask = (scores["split"] == "valid") & (scores["weight"] > 0)
        thr = weighted_quantile(scores.loc[vmask, f"p_{name}"].to_numpy(), scores.loc[vmask, "weight"].to_numpy(), 1 - ALERT_SHARE[name])
        model.meta["alert_threshold"] = thr
        model.meta["alert_share_on_validation"] = ALERT_SHARE[name]
        summary[name] = {
            "trees": model.meta["trees"],
            "terms": len(model.terms),
            "calibration": model.calibration,
            "alert_threshold": thr,
            "champion_coefficients": dict(zip(CHAMPION_FEATURES, champ[-1].coef_[0].round(4).tolist())),
            "benchmark_trees": int(bench.best_iteration),
        }
        tmask = (scores["split"] == "test") & (scores["weight"] > 0)
        s, w = scores[tmask], scores.loc[tmask, "weight"].to_numpy()
        for col in (f"p_{name}", f"champion_{name}", f"card_only_{name}", f"benchmark_{name}", f"restricted_{name}"):
            summary[name][f"test_auc_{col}"] = M.auc(s[y], s[col], w)
        log("  out-of-time AUC  " + "  ".join(f"{k.split('test_auc_')[1].rsplit('_' + name, 1)[0]}={v:.4f}" for k, v in summary[name].items() if k.startswith("test_auc_")))

    for name, model in models.items():
        store.write_json(f"models/{name}.json", model.to_json())
    scores.to_parquet(store.work / "scores.parquet", index=False)
    store.write_json("metrics/train_summary.json", summary)
    return summary
