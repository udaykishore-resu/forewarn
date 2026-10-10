"""Weighted performance, calibration and stability metrics.

Everything accepts sample weights because the risk models are evaluated on a
population where pilot-treated rows are removed and control rows are weighted up
(see features.build_labels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _w(y, w):
    y = np.asarray(y, dtype=np.float64)
    return y, (np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64))


def auc(y, s, w=None) -> float:
    """Weighted ROC AUC with tie handling (Mann-Whitney)."""
    y, w = _w(y, w)
    s = np.asarray(s, dtype=np.float64)
    keep = w > 0
    y, s, w = y[keep], s[keep], w[keep]
    order = np.argsort(s, kind="mergesort")
    y, s, w = y[order], s[order], w[order]
    wp, wn = w * y, w * (1 - y)
    # group ties
    new = np.r_[True, s[1:] != s[:-1]]
    g = np.cumsum(new) - 1
    gp = np.bincount(g, weights=wp)
    gn = np.bincount(g, weights=wn)
    cum_n_before = np.cumsum(gn) - gn
    num = np.sum(gp * (cum_n_before + 0.5 * gn))
    den = gp.sum() * gn.sum()
    return float(num / den) if den > 0 else float("nan")


def ks(y, s, w=None) -> float:
    y, w = _w(y, w)
    order = np.argsort(-np.asarray(s, dtype=np.float64), kind="mergesort")
    y, w = y[order], w[order]
    cp = np.cumsum(w * y) / max((w * y).sum(), 1e-12)
    cn = np.cumsum(w * (1 - y)) / max((w * (1 - y)).sum(), 1e-12)
    return float(np.max(np.abs(cp - cn)))


def capture(y, s, w=None, top=0.05, amount=None) -> dict:
    """What the top `top` share of accounts by score contains."""
    y, w = _w(y, w)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    k = int(np.searchsorted(cw, top, side="right"))
    sel = order[: max(k, 1)]
    out = {
        "top_share": top,
        "threshold": float(s[sel[-1]]),
        "event_capture": float((w[sel] * y[sel]).sum() / max((w * y).sum(), 1e-12)),
        "precision": float((w[sel] * y[sel]).sum() / w[sel].sum()),
        "base_rate": float((w * y).sum() / w.sum()),
    }
    out["lift"] = out["precision"] / max(out["base_rate"], 1e-12)
    if amount is not None:
        amount = np.asarray(amount, dtype=np.float64)
        out["amount_capture"] = float((w[sel] * amount[sel]).sum() / max((w * amount).sum(), 1e-12))
        out["amount_in_top"] = float((w[sel] * amount[sel]).sum())
    return out


def brier(y, p, w=None) -> float:
    y, w = _w(y, w)
    return float(np.sum(w * (np.asarray(p) - y) ** 2) / w.sum())


def calibration_table(y, p, w=None, bins=10) -> pd.DataFrame:
    y, w = _w(y, w)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    cw = np.cumsum(w[order]) / w.sum()
    b = np.minimum((cw * bins).astype(int), bins - 1)
    df = pd.DataFrame({"bin": b, "w": w[order], "wy": (w * y)[order], "wp": (w * p)[order]})
    g = df.groupby("bin").sum()
    out = pd.DataFrame(
        {
            "bin": g.index + 1,
            "weight": g["w"],
            "predicted": g["wp"] / g["w"],
            "observed": g["wy"] / g["w"],
        }
    )
    out["observed_over_predicted"] = out["observed"] / out["predicted"]
    return out.reset_index(drop=True)


def psi(ref_share: np.ndarray, cur_share: np.ndarray, eps: float = 1e-4) -> float:
    r = np.maximum(np.asarray(ref_share, dtype=np.float64), eps)
    c = np.maximum(np.asarray(cur_share, dtype=np.float64), eps)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def bin_shares(x, edges: np.ndarray, w=None) -> np.ndarray:
    """Share of weight per bin; the last bin is reserved for missing values.

    Bin i holds values in (edges[i-1], edges[i]], matching the scorer.
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=np.float64)
    miss = np.isnan(x)
    idx = np.searchsorted(edges, x[~miss], side="left")
    counts = np.bincount(idx, weights=w[~miss], minlength=len(edges) + 1)
    counts = np.r_[counts, w[miss].sum()]
    return counts / max(counts.sum(), 1e-12)


def quantile_edges(x, w=None, bins: int = 10) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    if w is not None:
        ok &= np.asarray(w) > 0
    q = np.unique(np.quantile(x[ok], np.linspace(0, 1, bins + 1)[1:-1])) if ok.any() else np.array([])
    return q.astype(np.float64)


def cluster_bootstrap(fn, groups: np.ndarray, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% interval for fn(multiplicity_per_row) resampling whole customers."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    vals = []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        v = fn(mult[inv].astype(np.float64))
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
