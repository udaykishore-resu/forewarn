import numpy as np
from sklearn.metrics import roc_auc_score

from forewarn import metrics as M


def test_weighted_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = np.round(rng.normal(y * 0.8, 1.0), 1)             # rounding creates many ties
    w = rng.integers(1, 6, 5000).astype(float)
    assert abs(M.auc(y, s, w) - roc_auc_score(y, s, sample_weight=w)) < 1e-12
    assert abs(M.auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_zero_weight_rows_are_ignored():
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8, 0.0])
    w = np.array([1, 1, 1, 1, 0.0])
    assert M.auc(y, s, w) == 1.0                           # the only mis-ranked row carries no weight


def test_capture_and_ks():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1).astype(float)
    c = M.capture(y, s, top=0.2, amount=y * 100.0)
    assert c["event_capture"] == 1.0 and c["precision"] == 1.0 and c["lift"] == 5.0 and c["amount_capture"] == 1.0
    assert M.ks(y, s) == 1.0


def test_psi_and_binning_rule():
    edges = np.array([1.0, 2.0])
    x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, np.nan])
    # bins are (lo, hi]: 1.0 falls in the first bin, 2.0 in the second; NaN has its own bin
    assert np.allclose(M.bin_shares(x, edges), np.array([2, 2, 1, 1]) / 6)
    ref = M.bin_shares(x, edges)
    assert M.psi(ref, ref) == 0.0
    assert M.psi(ref, np.array([0.7, 0.1, 0.1, 0.1])) > 0.25


def test_calibration_table_recovers_known_rates():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.5, 200_000)
    y = (rng.random(200_000) < p).astype(int)
    t = M.calibration_table(y, p)
    assert len(t) == 10 and (t["observed_over_predicted"].sub(1).abs() < 0.08).all()
    assert abs(M.brier(y, p) - np.mean(p * (1 - p))) < 0.003


def test_cluster_bootstrap_interval_contains_the_estimate():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(500), 4)
    y = rng.integers(0, 2, 2000)
    s = rng.normal(y, 1.0)
    lo, hi = M.cluster_bootstrap(lambda m: M.auc(y, s, m), g, n_boot=100)
    assert lo < M.auc(y, s) < hi and hi - lo < 0.1
