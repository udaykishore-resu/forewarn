"""The synthetic bank has to be internally consistent, or every metric built on it is noise."""
import numpy as np
import pandas as pd

from forewarn.features import make_panel
from forewarn.simulate import ARMS


def test_card_statements_reconcile_with_card_transactions(bank):
    """Balance this month = balance last month + everything posted to the card this month."""
    st, tx = bank["card_statements"], bank["card_transactions"]
    panel = make_panel(st)
    st = st.assign(m=panel.month_index(st["statement_date"]))
    tx = tx.assign(m=panel.month_index(tx["posted_date"]))
    posted = tx.groupby(["customer_id", "m"])["amount"].sum().rename("posted").reset_index()
    prev = st[["customer_id", "m", "statement_balance"]].assign(m=lambda d: d["m"] + 1).rename(columns={"statement_balance": "prev"})
    d = st.merge(prev, on=["customer_id", "m"]).merge(posted, on=["customer_id", "m"], how="left").fillna({"posted": 0.0})
    # a balance cannot go below zero, so an overpayment is absorbed
    expected = np.maximum(d["prev"] + d["posted"], 0.0)
    assert len(d) > 100_000
    assert np.abs(expected - d["statement_balance"]).max() < 0.06      # cents of rounding per posted row


def test_deposit_ledger_is_dated_inside_its_month_and_signed_by_category(bank):
    tx = bank["dda_transactions"]
    inflow = {"payroll", "gov_benefit", "other_inflow", "transfer_in_external", "shortterm_loan_proceeds", "cash_advance_deposit"}
    is_in = tx["category"].astype(str).isin(inflow)
    assert (tx.loc[is_in, "amount"] > 0).all()
    assert (tx.loc[~is_in, "amount"] < 0).all()
    assert tx["posted_date"].dt.day.max() <= 28


def test_outcomes_are_consistent_with_statements(bank):
    st, oc = bank["card_statements"], bank["outcomes"]
    last = st.sort_values("statement_date").groupby("customer_id").tail(1).set_index("customer_id")
    co = oc[oc["chargeoff_date"].notna()].set_index("customer_id")
    bk = oc[oc["bankruptcy_filing_date"].notna()].set_index("customer_id")
    assert len(co) > 20 and len(bk) > 20
    assert (last.loc[co.index, "status"].astype(str) == "charged_off").all()
    assert (last.loc[bk.index, "status"].astype(str) == "bankrupt").all()
    assert (last.loc[co.index, "statement_date"] == co["chargeoff_date"]).all()      # nothing is recorded after closure
    assert (last.loc[co.index, "cycles_past_due"] >= 6).all()
    # credit loss is booked once, on the closing statement, and matches the outcome table
    loss = st.groupby("customer_id")["credit_loss"].sum()
    assert np.allclose(loss.reindex(oc["customer_id"]).to_numpy(), oc["loss_amount"].to_numpy(), atol=0.02)
    good = oc[oc["chargeoff_date"].isna() & oc["bankruptcy_filing_date"].isna()]
    assert (good["loss_amount"] == 0).all()


def test_delinquency_moves_one_cycle_at_a_time(bank):
    st = bank["card_statements"].sort_values(["customer_id", "statement_date"])
    step = st.groupby("customer_id")["cycles_past_due"].diff().dropna()
    assert step.max() <= 1                               # 0 -> 2 in one month would be a bug; cures can drop by any amount


def test_pilot_is_randomized(bank):
    iv = bank["interventions"]
    assert iv["customer_id"].is_unique
    counts = iv["arm"].value_counts().reindex(ARMS)
    expected = len(iv) / len(ARMS)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 18.5                                   # p > 0.001 with 4 degrees of freedom
    # arm must not depend on anything about the customer: compare prior utilization across arms
    st = bank["card_statements"].merge(iv[["customer_id", "decision_date", "arm"]], on="customer_id")
    at = st[st["statement_date"] == st["decision_date"]]
    util = (at["statement_balance"] / at["credit_limit"]).groupby(at["arm"].astype(str)).mean()
    assert util.max() - util.min() < 0.04


def test_protected_attributes_are_not_in_any_model_input_table(bank):
    for name in ("customers", "dda_transactions", "card_transactions", "card_statements", "outcomes", "interventions"):
        assert not {"group_a", "group_b"} & set(bank[name].columns), name
