"""Assemble the model bundle the scorer loads, and the parity cases that prove
the scorer reproduces the reference implementation."""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import pandas as pd

from . import __version__, metrics as M
from .features import ARMS
from .recommend import ELIGIBILITY, PILOT_SUPPORT, VALUE_PARAMS
from .registry import assert_permitted, registry_json, value_features
from .score import decide, row_view
from .store import Store, file_sha256
from .train import load_dataset

SCHEMA_VERSION = 1
REASONS = {"max_reasons": 4, "min_points": 0.02}
ADVERSE_ACTIONS = ["limit_decrease"]


def build_bundle(store: Store, data_source: str = "synthetic") -> dict:
    df = load_dataset(store)
    models = {name: store.read_json(f"models/{name}.json") for name in ("delinquency", "bankruptcy")}
    models["value_effects"] = {a: store.read_json(f"models/value_effect_{a}.json") for a in ARMS}
    used = sorted(
        {t["feature"] for m in (models["delinquency"], models["bankruptcy"]) for t in m["terms"]}
        | {t["feature"] for m in models["value_effects"].values() for t in m["terms"]}
        | {c["feature"] for cs in ELIGIBILITY.values() for c in cs}
        | {c["feature"] for clause in PILOT_SUPPORT for c in clause}
    )
    # Every permitted feature is part of the input contract, used or not, so a bank's
    # feature pipeline does not change when a retrained model picks different inputs.
    order = value_features()
    assert_permitted(order)
    assert set(used) <= set(order), sorted(set(used) - set(order))

    ref = df[(df["split"] == "train") & (df["weight"] > 0)]
    w = ref["weight"].to_numpy()
    drift = {"features": {}, "scores": {}}
    features = registry_json(order)
    for spec in features:
        f = spec["name"]
        x = ref[f].to_numpy(dtype=np.float64)
        edges = M.quantile_edges(x, w)
        drift["features"][f] = {"edges": edges.tolist(), "shares": M.bin_shares(x, edges, w).tolist()}
        # A feature that was never missing in training has no learned missing-value
        # behaviour, so the scorer must refuse to guess one.
        spec["required"] = bool(not np.isnan(x).any())
    scores = pd.read_parquet(store.work / "scores.parquet")
    sref = scores[(scores["split"] == "train") & (scores["weight"] > 0)]
    alerts = {}
    for name in ("delinquency", "bankruptcy"):
        m = models[name]
        a, b = m["calibration"]["a"], m["calibration"]["b"]
        z = a + b * sref[f"raw_{name}"].to_numpy()
        edges = M.quantile_edges(z, sref["weight"].to_numpy())
        drift["scores"][name] = {"edges": edges.tolist(), "shares": M.bin_shares(z, edges, sref["weight"].to_numpy()).tolist()}
        thr = m["meta"]["alert_threshold"]
        alerts[name] = {
            "threshold_probability": thr,
            "threshold_logit": math.log(thr / (1 - thr)),
            "share_alerted_on_validation": m["meta"]["alert_share_on_validation"],
        }

    summary = store.read_json("metrics/train_summary.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "forewarn",
        "version": __version__,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "data_source": data_source,
        "training": {
            "rows": summary["rows"],
            "features_sha256": file_sha256(store.work / "features.parquet"),
            "snapshot_months": {
                s: [int(df.loc[df["split"] == s, "snapshot_month"].min()), int(df.loc[df["split"] == s, "snapshot_month"].max())]
                for s in ("train", "valid", "test")
            },
            "targets": {
                "delinquency": "60 or more days past due within 6 months, account current at snapshot",
                "bankruptcy": "bankruptcy filing within 6 months, account current at snapshot",
            },
        },
        "features": features,
        "models": models,
        "alerts": alerts,
        "reasons": REASONS,
        "policy": {
            "arms": ARMS,
            "eligibility": ELIGIBILITY,
            "pilot_support": PILOT_SUPPORT,
            "min_gain": VALUE_PARAMS["min_gain"],
            "adverse_actions": ADVERSE_ACTIONS,
            "value_horizon_months": 18,
        },
        "drift_reference": drift,
    }


def export(store: Store, n_cases: int = 400, seed: int = 0, log=print) -> dict:
    bundle = build_bundle(store)
    out_dir = store.artifacts / "bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bundle.json"
    path.write_text(json.dumps(bundle, indent=1) + "\n")
    digest = file_sha256(path)
    (out_dir / "bundle.sha256").write_text(digest + "\n")

    # parity cases: random accounts, alerted accounts, and accounts with missing deposit data
    df = load_dataset(store)
    te = df[df["split"] == "test"]
    names = [f["name"] for f in bundle["features"]]
    d_all = decide(bundle, te[names], with_reasons=False)
    rng = np.random.default_rng(seed)
    idx_alert = np.flatnonzero(d_all["alert"])
    idx_rec = np.flatnonzero(d_all["action"]["recommended"] != "none")
    idx_miss = np.flatnonzero(te["buffer_days"].isna().to_numpy())
    k = n_cases // 4
    pick = np.unique(np.concatenate([
        rng.choice(len(te), k, replace=False),
        rng.choice(idx_alert, min(k, len(idx_alert)), replace=False),
        rng.choice(idx_rec, min(k, len(idx_rec)), replace=False),
        rng.choice(idx_miss, min(k, len(idx_miss)), replace=False),
    ]))
    Xc = te.iloc[pick][names].reset_index(drop=True)
    d = decide(bundle, Xc)
    example = None
    with open(out_dir / "parity_cases.jsonl", "w") as fh:
        for i in range(len(Xc)):
            feats = {n: (None if pd.isna(v) else float(v)) for n, v in Xc.iloc[i].items()}
            fh.write(json.dumps({"features": feats, "expect": row_view(bundle, d, i)}) + "\n")
            if example is None and d["action"]["recommended"][i] != "none" and not Xc.iloc[i].isna().any():
                example = {"account_ref": "example-1", "as_of": "2026-09-30",
                           "features": {k: round(v, 4) for k, v in feats.items()}}
    if example is not None:                      # a complete, alerted account to try the scorer with
        (out_dir / "example_request.json").write_text(json.dumps(example, indent=1) + "\n")
    info = {
        "bundle": str(path), "sha256": digest, "features": len(names), "parity_cases": int(len(Xc)),
        "bytes": path.stat().st_size,
        "test_alert_rate": float(np.average(d_all["alert"], weights=te["weight"])),
        "alerted_in_pilot_support": float(d_all["action"]["in_pilot_support"][d_all["alert"]].mean()),
        "alerted_action_mix": pd.Series(d_all["action"]["recommended"][d_all["alert"]]).value_counts(normalize=True).round(4).to_dict(),
    }
    store.write_json("metrics/export.json", info)
    log(f"bundle {digest[:12]}  {info['bytes'] / 1024:.0f} KiB  {len(names)} features  {len(Xc)} parity cases")
    log(f"  alerted accounts in pilot support: {info['alerted_in_pilot_support']:.1%}; action mix among alerted: {info['alerted_action_mix']}")
    return info
