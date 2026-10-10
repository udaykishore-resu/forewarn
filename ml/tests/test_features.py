"""Feature definitions checked against a ledger small enough to work out by hand."""
import numpy as np
import pandas as pd
import pytest

from forewarn.features import build_features, build_labels
from forewarn.registry import FEATURES, assert_permitted, model_features
from forewarn.simulate import CARD_CATEGORIES, DDA_CATEGORIES


def _bank(months=12):
    """One customer, twelve months.  Paid on the 15th and 28th until payroll stops after month 8."""
    dates = pd.date_range("2024-01-31", periods=months, freq="ME")
    dda, card, stm = [], [], []
    for m, end in enumerate(dates):
        first = end.replace(day=1)
        if m <= 8:
            dda += [(first.replace(day=15), 2000.0, "payroll"), (first.replace(day=28), 2000.0, "payroll")]
        dda += [(first.replace(day=1), -1500.0, "housing"), (first.replace(day=10), -1500.0, "essential_debit")]
        if m >= 10:
            dda.append((first.replace(day=28), -34.0, "fee_overdraft"))
        if m == 11:
            card.append((first.replace(day=5), 300.0, "cash_advance"))
        card.append((first.replace(day=14), 400.0, "purchase_essential"))
        bal = 1000.0 + 200.0 * m
        stm.append((end, bal, 5000.0, 35.0, 35.0 if m >= 9 else 400.0, 400.0))
    customers = pd.DataFrame({"customer_id": [7], "has_dda": [True], "dda_opening_balance": [3000.0]})
    d = pd.DataFrame(dda, columns=["posted_date", "amount", "category"]).assign(customer_id=7)
    d["category"] = pd.Categorical(d["category"], DDA_CATEGORIES)
    c = pd.DataFrame(card, columns=["posted_date", "amount", "category"]).assign(customer_id=7)
    c["category"] = pd.Categorical(c["category"], CARD_CATEGORIES)
    s = pd.DataFrame(stm, columns=["statement_date", "statement_balance", "credit_limit", "min_due", "payments", "purchases"]).assign(
        customer_id=7, cycles_past_due=0, status=pd.Categorical(["open"] * months, ["open", "charged_off", "bankrupt", "closed"]),
        apr=0.24, due_day=20, revenue=0.0, credit_loss=0.0, penalty_fees=0.0)
    return customers, d, c, s


def test_hand_worked_values():
    customers, d, c, s = _bank()
    snap, _ = build_features(customers, d, c, s)
    row = snap.set_index("snapshot_month")

    # month 8 (September): last payroll on the 28th, the month ends on the 30th
    assert row.loc[8, "payroll_gap_days"] == 0                    # 2 days: inside a normal cycle
    assert row.loc[8, "payroll_missed_cycles_3m"] == 0
    assert row.loc[8, "payroll_amt_ratio_3m"] == 1.0
    assert row.loc[8, "od_count_3m"] == 0

    # month 11 (December): payroll stopped after September 28 -> 94 days; three months with no cheques
    assert row.loc[11, "payroll_gap_days"] == 94 - 31
    assert row.loc[9, "payroll_gap_days"] == 33 - 31              # October 31: last cheque September 28
    assert row.loc[11, "payroll_amt_ratio_3m"] == 0.0
    assert row.loc[11, "payroll_missed_cycles_3m"] == 6           # six cheques expected in three months, none arrived
    assert row.loc[11, "od_count_3m"] == 2 and row.loc[11, "od_months_6m"] == 2
    assert row.loc[11, "cash_adv_months_6m"] == 1
    assert row.loc[11, "cash_adv_to_limit_3m"] == pytest.approx(300 / 5000, abs=1e-4)
    assert row.loc[11, "utilization"] == pytest.approx((1000 + 200 * 11) / 5000, abs=1e-4)
    assert row.loc[11, "util_change_3m"] == pytest.approx(600 / 5000, abs=1e-4)
    assert row.loc[11, "min_pay_months_6m"] == 3                  # months 9, 10, 11 paid exactly the minimum
    assert row.loc[11, "open_to_buy"] == 5000 - 3200

    # balance: 3000 opening + 1000 a month for nine months, then -3000 a month, less two overdraft fees
    expected_balance = 3000 + 9 * 1000 - 3 * 3000 - 2 * 34
    outflow_per_day = (3 * 3000 + 2 * 34) / 90
    assert row.loc[11, "buffer_days"] == pytest.approx(expected_balance / outflow_per_day, abs=1e-3)


def test_payroll_older_than_120_days_is_treated_as_no_payroll_relationship():
    customers, d, c, s = _bank(months=14)
    snap, _ = build_features(customers, d, c, s)
    row = snap.set_index("snapshot_month")
    assert row.loc[11, "payroll_gap_days"] == 63                   # 94 days: still inside the window
    assert np.isnan(row.loc[12, "payroll_gap_days"])               # 125 days: no longer a "gap"
    assert np.isnan(row.loc[13, "payroll_gap_days"])


def test_customers_without_a_deposit_account_have_missing_deposit_features():
    customers, d, c, s = _bank()
    customers["has_dda"] = False
    snap, _ = build_features(customers, d.iloc[:0], c, s)
    for f in FEATURES:
        if f.source == "deposit" and f.status != "prohibited":
            assert snap[f.name].isna().all(), f.name
    assert snap["utilization"].notna().all()


def test_prohibited_and_restricted_features_stay_out(snapshots):
    snap, _, _ = snapshots
    assert "gov_benefit_flag_3m" not in snap.columns               # never computed
    assert "legal_services_3m" not in model_features()             # computed, not used
    assert "gov_benefit_flag_3m" not in model_features(include_restricted=True)
    with pytest.raises(ValueError, match="prohibited"):
        assert_permitted(["gov_benefit_flag_3m"], allow_restricted=True)
    with pytest.raises(ValueError, match="restricted"):
        assert_permitted(["legal_services_3m"])
    with pytest.raises(ValueError, match="not in the registry"):
        assert_permitted(["zip_code"])


def test_labels_look_forward_six_months_and_weights_remove_treated_rows(snapshots):
    snap, lab, panel = snapshots
    assert set(lab["y_dlq"].unique()) <= {0, 1}
    assert lab.loc[lab["observed_6m"], "snapshot_month"].max() == panel.n_months - 1 - 6
    treated = ~lab["arm"].isin(["not_enrolled", "none"])
    after = lab["snapshot_month"] + 6 >= lab["decision_month"]
    assert (lab.loc[treated & after, "weight"] == 0).all()         # outcomes changed by an action never train a risk model
    assert (lab.loc[treated & ~after, "weight"] == 1).all()
    assert (lab.loc[(lab["arm"] == "none") & after, "weight"] == 5).all()   # controls stand in for their stratum
    assert (lab.loc[lab["arm"] == "not_enrolled", "weight"] == 1).all()
    d = lab[lab["is_decision_row"]]
    assert d["customer_id"].is_unique and (d["snapshot_month"] == d["decision_month"]).all()


def test_label_definition_on_a_known_account(bank, snapshots):
    """Pick an account that reached 60+ days past due and check the label flips exactly six months before."""
    snap, lab, panel = snapshots
    st = bank["card_statements"].assign(m=lambda x: panel.month_index(x["statement_date"]))
    first60 = st[st["cycles_past_due"] >= 2].groupby("customer_id")["m"].min()
    first60 = first60[(first60 >= 20) & (first60 <= 40)]
    checked = 0
    for cust, m60 in first60.items():
        rows = lab[(lab["customer_id"] == cust) & (lab["snapshot_month"] < m60)].set_index("snapshot_month")["y_dlq"]
        earlier = st[(st["customer_id"] == cust) & (st["m"] < m60) & (st["m"] >= m60 - 12)]
        if (earlier["cycles_past_due"] >= 2).any() or len(rows) < 8:
            continue
        for t, y in rows.items():
            if m60 - 6 <= t:
                assert y == 1, (cust, t, m60)
            elif t < m60 - 6 and t >= m60 - 12:
                assert y == 0, (cust, t, m60)
        checked += 1
        if checked == 25:
            break
    assert checked >= 10
