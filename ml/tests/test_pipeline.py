"""End to end on a small simulated bank: train, recommend, export, then check the
bundle, the reference decisions, and (when Go is installed) the Go scorer."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forewarn.export import export
from forewarn.gam import AdditiveModel
from forewarn.recommend import evaluate
from forewarn.score import decide, row_view
from forewarn.store import Store, file_sha256
from forewarn.train import load_dataset, train

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def built(bank, snapshots, tmp_path_factory):
    snap, lab, _ = snapshots
    store = Store(tmp_path_factory.mktemp("forewarn"))
    store.write_raw(bank)
    snap.to_parquet(store.work / "features.parquet", index=False)
    lab.to_parquet(store.work / "labels.parquet", index=False)
    quiet = lambda *_: None  # noqa: E731
    train(store, log=quiet)
    policy = evaluate(store, n_boot=200, log=quiet)
    info = export(store, n_cases=200, log=quiet)
    bundle = json.loads((store.artifacts / "bundle" / "bundle.json").read_text())
    return store, bundle, policy, info


def test_bundle_contains_only_permitted_features_and_is_pinned(built):
    store, bundle, _, info = built
    assert {f["status"] for f in bundle["features"]} == {"allowed"}
    names = {f["name"] for f in bundle["features"]}
    assert "gov_benefit_flag_3m" not in names and "legal_services_3m" not in names
    assert file_sha256(store.artifacts / "bundle" / "bundle.json") == info["sha256"]
    assert (store.artifacts / "bundle" / "bundle.sha256").read_text().strip() == info["sha256"]
    assert bundle["data_source"] == "synthetic"


def test_reference_decisions_agree_with_the_models(built):
    store, bundle, _, _ = built
    df = load_dataset(store)
    X = df[df["split"] == "test"].sample(3000, random_state=0)
    d = decide(bundle, X, with_reasons=False)
    for name in ("delinquency", "bankruptcy"):
        model = AdditiveModel.from_json(bundle["models"][name])
        assert np.allclose(d[name]["probability"], model.predict(X), rtol=0, atol=1e-12)
        assert np.array_equal(d[name]["alert"], model.predict(X) >= bundle["alerts"][name]["threshold_probability"])
    a = d["action"]
    rec = a["recommended"]
    assert (rec[~d["alert"]] == "none").all()                              # no alert, no recommendation
    assert (rec[~a["in_pilot_support"]] == "none").all()                   # no recommendation outside the pilot's population
    arms = bundle["policy"]["arms"]
    for j, arm in enumerate(arms[1:], start=1):
        chosen = rec == arm
        assert a["eligible"][chosen, j].all()                              # never an ineligible action
        assert (a["gain"][chosen, j] >= bundle["policy"]["min_gain"]).all()
    assert np.array_equal(a["adverse_action_notice"], rec == "limit_decrease")


def test_reasons_come_from_the_score_and_respect_the_limits(built):
    store, bundle, _, _ = built
    df = load_dataset(store)
    te = df[df["split"] == "test"]
    names = [f["name"] for f in bundle["features"]]
    group = {f["name"]: f["group"] for f in bundle["features"]}
    probe = decide(bundle, te[names], with_reasons=False)
    X = te[probe["delinquency"]["alert"]].head(300)[names].reset_index(drop=True)
    d = decide(bundle, X)
    m = bundle["models"]["delinquency"]
    for i in range(len(X)):
        rs = d["delinquency"]["reasons"][i]
        assert 1 <= len(rs) <= bundle["reasons"]["max_reasons"]
        assert len({group[r["feature"]] for r in rs}) == len(rs)           # one reason per group
        pts = [r["points"] for r in rs]
        assert pts == sorted(pts, reverse=True) and min(pts) > bundle["reasons"]["min_points"]
        # points are the group's summed contributions, straight from the tables
        contrib = dict(zip([t["feature"] for t in m["terms"]], d["delinquency"]["contributions"][i]))
        for r in rs:
            total = sum(c for f, c in contrib.items() if group[f] == group[r["feature"]])
            assert abs(total - r["points"]) < 1e-12
            assert r["code"].endswith("M") == bool(pd.isna(X.loc[i, r["feature"]]))
    view = row_view(bundle, d, 0)
    assert set(view) == {"alert", "driver", "scores", "action"} and view["alert"] is True


def test_required_features_cannot_be_missing(built):
    store, bundle, _, _ = built
    required = [f["name"] for f in bundle["features"] if f["required"]]
    optional = [f["name"] for f in bundle["features"] if not f["required"]]
    assert "utilization" in required and "buffer_days" in optional
    df = load_dataset(store)
    X = df[df["split"] == "test"].head(50).copy()
    X.loc[X.index[3], "utilization"] = np.nan
    with pytest.raises(ValueError, match="required"):
        decide(bundle, X, with_reasons=False)
    example = json.loads((store.artifacts / "bundle" / "example_request.json").read_text())
    assert set(example["features"]) == {f["name"] for f in bundle["features"]}


def test_policy_evaluation_is_well_formed(built):
    _, _, policy, _ = built
    assert set(policy["arms"]) == {"none", "due_date_change", "hardship_plan", "limit_decrease", "outreach_call"}
    assert policy["policies"]["no_action"]["gain_vs_no_action"] == 0.0
    fw = policy["policies"]["forewarn_policy"]
    assert fw["gain_ci95"][0] <= fw["gain_vs_no_action"] <= fw["gain_ci95"][1]
    assert abs(sum(fw["action_mix"].values()) - 1) < 1e-3


@pytest.mark.skipif(shutil.which("go") is None, reason="Go toolchain not installed")
def test_go_scorer_reproduces_the_reference_cases(built):
    store, _, _, info = built
    out = subprocess.run(
        ["go", "run", "./cmd/forewarn", "verify", "-bundle", str(store.artifacts / "bundle" / "bundle.json"),
         "-bundle-sha256", info["sha256"], "-cases", str(store.artifacts / "bundle" / "parity_cases.jsonl")],
        cwd=REPO / "scorer", capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    rep = json.loads(out.stdout)
    assert rep["failed"] == 0 and rep["passed"] == rep["cases"] >= 100 and rep["max_abs_diff_logit"] < 1e-12
