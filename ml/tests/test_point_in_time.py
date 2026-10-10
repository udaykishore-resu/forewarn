"""A feature for month t may use nothing dated after the last day of month t.

The check is direct: cut every input table off at a date, rebuild the features,
and require the rows for that month to be identical to the ones built from the
full history.  Any look-ahead, however indirect, changes at least one value.
"""
import numpy as np
import pandas as pd
import pytest

from forewarn.features import build_features
from forewarn.registry import FEATURES, VALUE_EXTRAS


@pytest.mark.parametrize("month", [9, 23, 40])
def test_features_do_not_change_when_the_future_is_removed(bank, snapshots, month):
    full, _, panel = snapshots
    as_of = pd.Timestamp(panel.month_end_date(month))
    cut, cut_panel = build_features(bank["customers"], bank["dda_transactions"], bank["card_transactions"], bank["card_statements"], as_of=as_of)
    assert cut_panel.n_months == month + 1
    a = full[full["snapshot_month"] == month].set_index("customer_id").sort_index()
    b = cut[cut["snapshot_month"] == month].set_index("customer_id").sort_index()
    assert len(a) > 1000 and a.index.equals(b.index)
    for f in FEATURES + VALUE_EXTRAS:
        if f.status == "prohibited":
            continue
        x, y = a[f.name].to_numpy(dtype=np.float64), b[f.name].to_numpy(dtype=np.float64)
        assert np.array_equal(np.isnan(x), np.isnan(y)), f.name
        assert np.allclose(x[~np.isnan(x)], y[~np.isnan(y)], rtol=0, atol=1e-9), f.name


def test_snapshots_only_cover_open_current_accounts(bank, snapshots):
    snap, _, panel = snapshots
    st = bank["card_statements"].assign(snapshot_month=lambda d: panel.month_index(d["statement_date"]))
    j = snap.merge(st, on=["customer_id", "snapshot_month"], how="left")
    assert (j["cycles_past_due"] == 0).all()
    assert (j["status"].astype(str) == "open").all()
    assert snap["snapshot_month"].min() == 8             # nine months of history before the first score
