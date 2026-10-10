"""Point-in-time features from the bank's own deposit and card data.

Everything is computed as of the last day of a snapshot month using only rows
dated on or before that day.  `tests/test_point_in_time.py` proves this by
truncating the inputs and checking the features do not change.

Inputs follow docs/data-contract.md.  No table from another institution, no
bureau file and no open-banking feed is required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .registry import FEATURES, VALUE_EXTRAS

FEATURE_DECIMALS = 4        # every non-dollar feature is rounded to this many places
PAYROLL_CYCLE_DAYS = 31     # the longest normal gap between cheques
PAYROLL_STALE_DAYS = 120    # after this the payroll relationship is treated as ended
FIRST_SNAPSHOT = 8          # months of history needed (3 recent + 6 prior, zero-based)
HORIZON = 6                 # outcome window in months
ARMS = ["none", "due_date_change", "hardship_plan", "limit_decrease", "outreach_call"]


@dataclass
class Panel:
    base: np.datetime64          # first calendar month, datetime64[M]
    n_months: int
    customer_ids: np.ndarray     # sorted

    def month_index(self, dates) -> np.ndarray:
        d = np.asarray(pd.to_datetime(dates)).astype("datetime64[M]")
        return (d - self.base).astype(np.int64)

    def month_end_day(self) -> np.ndarray:
        """Absolute day number of the last day of every month."""
        ends = (self.base + np.arange(1, self.n_months + 1).astype("timedelta64[M]")).astype("datetime64[D]") - np.timedelta64(1, "D")
        return ends.astype(np.int64)

    def month_end_date(self, m) -> np.ndarray:
        nxt = self.base + (np.asarray(m, dtype=np.int64) + 1).astype("timedelta64[M]")
        return (nxt.astype("datetime64[D]") - np.timedelta64(1, "D")).astype("datetime64[s]")


def make_panel(statements: pd.DataFrame, customer_ids=None) -> Panel:
    d = np.asarray(pd.to_datetime(statements["statement_date"])).astype("datetime64[M]")
    base, last = d.min(), d.max()
    ids = np.sort(statements["customer_id"].unique()) if customer_ids is None else np.sort(np.asarray(customer_ids))
    return Panel(base, int((last - base).astype(np.int64)) + 1, ids)


# ----------------------------------------------------------------------------- helpers
def _rsum(a: np.ndarray, w: int) -> np.ndarray:
    """Sum over the trailing w months, inclusive of the current month."""
    c = np.cumsum(np.nan_to_num(a.astype(np.float64)), axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def _lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    if k < a.shape[1]:
        out[:, k:] = a[:, :-k]
    return out


def _rmax(a: np.ndarray, w: int) -> np.ndarray:
    out = a.astype(np.float64).copy()
    for k in range(1, w):
        out = np.fmax(out, _lag(a, k))
    return out


def _rmin(a: np.ndarray, w: int) -> np.ndarray:
    out = a.astype(np.float64).copy()
    for k in range(1, w):
        out = np.fmin(out, _lag(a, k))
    return out


def _ratio(num, den, floor, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den >= floor, num / den, np.nan)
    return np.clip(r, lo, hi)


def _ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for t in range(1, out.shape[1]):
        m = np.isnan(out[:, t])
        out[m, t] = out[m, t - 1]
    return out


def _grid(panel: Panel, cell: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    """Sum (or count) rows into an (N, T) array; `cell` is customer_index * T + month."""
    N, T = len(panel.customer_ids), panel.n_months
    return np.bincount(cell, weights=values, minlength=N * T).reshape(N, T).astype(np.float32)


def _by_category(panel: Panel, cust, month, amount, cat_codes, names, want, count=()):
    """Monthly sums (and optionally counts) per transaction category."""
    T = panel.n_months
    ok = (month >= 0) & (month < T)
    cell = (np.searchsorted(panel.customer_ids, cust[ok]) * T + month[ok]).astype(np.int64)
    amount, cat_codes = amount[ok], cat_codes[ok]
    sums, counts = {}, {}
    for name in want:
        m = cat_codes == names.index(name)
        sums[name] = _grid(panel, cell[m], amount[m])
        if name in count:
            counts[name] = _grid(panel, cell[m])
    return sums, counts


# ----------------------------------------------------------------------------- main
def build_features(
    customers: pd.DataFrame,
    dda: pd.DataFrame,
    card: pd.DataFrame,
    statements: pd.DataFrame,
    as_of=None,
    chunk_customers: int = 20_000,
) -> tuple[pd.DataFrame, Panel]:
    """Return one row per (customer, snapshot month) for accounts that are open and current.

    Features are per customer, so the work is done in customer chunks to bound
    memory; the result is identical to a single pass.
    """
    if as_of is not None:
        cut = pd.Timestamp(as_of)
        dda = dda[dda["posted_date"] <= cut]
        card = card[card["posted_date"] <= cut]
        statements = statements[statements["statement_date"] <= cut]
    full = make_panel(statements)
    ids = full.customer_ids
    parts = []
    for lo in range(0, len(ids), chunk_customers):
        a, b = ids[lo], ids[min(lo + chunk_customers, len(ids)) - 1]
        pick = lambda df: df[df["customer_id"].between(a, b)]  # noqa: E731
        panel = Panel(full.base, full.n_months, ids[lo:lo + chunk_customers])
        parts.append(_build_chunk(pick(customers), pick(dda), pick(card), pick(statements), panel))
    return pd.concat(parts, ignore_index=True), full


def _build_chunk(customers, dda, card, statements, panel: Panel) -> pd.DataFrame:
    N, T = len(panel.customer_ids), panel.n_months
    nan = np.full((N, T), np.nan)

    # ------------------------------------------------------------- card statements
    sc = statements["customer_id"].to_numpy()
    sm = panel.month_index(statements["statement_date"])
    si = np.searchsorted(panel.customer_ids, sc)

    def stmt(col, fill=np.nan, dtype=np.float64):
        a = np.full((N, T), fill, dtype)
        a[si, sm] = statements[col].to_numpy()
        return a

    bal, limit = stmt("statement_balance"), stmt("credit_limit")
    min_due, pay, purch = stmt("min_due"), stmt("payments"), stmt("purchases")
    cpd = stmt("cycles_past_due")
    is_open = np.zeros((N, T), bool)
    is_open[si, sm] = (statements["status"].astype(str) == "open").to_numpy()

    # ------------------------------------------------------------- card transactions
    cc = card["category"]
    C, Cn = _by_category(
        panel, card["customer_id"].to_numpy(), panel.month_index(card["posted_date"]),
        card["amount"].to_numpy(), cc.cat.codes.to_numpy(), list(cc.cat.categories),
        want=["purchase_discretionary", "cash_advance", "fee_late"], count=["fee_late"],
    )

    # ------------------------------------------------------------- deposit transactions
    has_dda = np.zeros(N, bool)
    opening = np.zeros(N)
    cust_i = np.searchsorted(panel.customer_ids, customers["customer_id"].to_numpy())
    keep = np.isin(customers["customer_id"].to_numpy(), panel.customer_ids)
    has_dda[cust_i[keep]] = customers["has_dda"].to_numpy()[keep]
    opening[cust_i[keep]] = np.nan_to_num(customers["dda_opening_balance"].to_numpy()[keep])

    dcat_s = dda["category"]
    dnames = list(dcat_s.cat.categories)
    dcust = dda["customer_id"].to_numpy()
    ddate = np.asarray(dda["posted_date"]).astype("datetime64[D]")
    dday = ddate.astype(np.int64)
    dm = panel.month_index(ddate)
    damt = dda["amount"].to_numpy()
    dcat = dcat_s.cat.codes.to_numpy().astype(np.int64)
    D, Dn = _by_category(panel, dcust, dm, damt, dcat, dnames, want=dnames, count=["payroll", "fee_overdraft"])
    is_payroll = dcat == dnames.index("payroll")
    payroll_n, od_n = Dn["payroll"], Dn["fee_overdraft"]

    # end-of-day balances -> monthly minimum, month-end balance, negative days
    order = np.lexsort((dday, dcust))
    oc, od_, om, oa = dcust[order], dday[order], dm[order], damt[order]
    oi = np.searchsorted(panel.customer_ids, oc)
    # running balances are kept in integer cents: a balance of exactly zero must not
    # become -1e-12 and count as a negative day
    cents = np.rint(oa * 100.0).astype(np.int64)
    csum = np.cumsum(cents)
    first = np.r_[True, oc[1:] != oc[:-1]]
    first_row = np.maximum.accumulate(np.where(first, np.arange(len(oa)), 0))
    start_c = (csum - cents)[first_row]
    eod_bal = (csum - start_c + np.rint(opening[oi] * 100.0).astype(np.int64)) / 100.0
    is_eod = np.r_[(oc[1:] != oc[:-1]) | (od_[1:] != od_[:-1]), True] if len(oa) else np.zeros(0, bool)
    ei, em, eb = oi[is_eod], om[is_eod], eod_bal[is_eod]
    okm = (em >= 0) & (em < T)
    ei, em, eb = ei[okm], em[okm], eb[okm]
    min_bal = nan.copy()
    eom_bal = nan.copy()
    neg_days = np.zeros((N, T))
    if len(eb):
        key = ei * T + em
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        ends = np.r_[starts[1:], len(key)] - 1
        gi, gm = ei[starts], em[starts]
        min_bal[gi, gm] = np.minimum.reduceat(eb, starts)
        eom_bal[gi, gm] = eb[ends]
        neg_days[gi, gm] = np.add.reduceat((eb < 0).astype(np.float64), starts)
    # months without activity carry the previous balance
    seed = eom_bal.copy()
    seed[:, 0] = np.where(np.isnan(seed[:, 0]), opening, seed[:, 0])
    eom_bal = _ffill(seed)
    min_bal = np.where(np.isnan(min_bal), eom_bal, min_bal)

    # last payroll date, carried forward
    last_pay = nan.copy()
    if is_payroll.any():
        pi = np.searchsorted(panel.customer_ids, dcust[is_payroll])
        pm, pdv = dm[is_payroll], dday[is_payroll].astype(np.float64)
        okp = (pm >= 0) & (pm < T)
        flat = np.full(N * T, -np.inf)
        np.maximum.at(flat, pi[okp] * T + pm[okp], pdv[okp])
        last_pay = np.where(np.isfinite(flat), flat, np.nan).reshape(N, T)
    last_pay = _ffill(last_pay)
    month_end = panel.month_end_day()[None, :].astype(np.float64)

    # ------------------------------------------------------------- deposit features
    f: dict[str, np.ndarray] = {}
    income_like = D["payroll"] + D["gov_benefit"] + D["other_inflow"]
    inflow_all = income_like + D["transfer_in_external"] + D["shortterm_loan_proceeds"] + D["cash_advance_deposit"]
    outflow = -(
        D["housing"] + D["essential_debit"] + D["discretionary_debit"] + D["loan_payment_external"]
        + D["loan_payment_shortterm"] + D["card_payment_internal"] + D["large_one_off"]
        + D["legal_services"] + D["fee_overdraft"]
    )
    out3 = _rsum(outflow, 3)

    since = month_end - last_pay
    # Counted beyond a normal monthly cycle.  Raw days-since-payroll swings with month
    # length and pay calendars, which shows up as input drift every single month.
    f["payroll_gap_days"] = np.where(since <= PAYROLL_STALE_DAYS, np.maximum(since - PAYROLL_CYCLE_DAYS, 0.0), np.nan)
    prior_pay = _rsum(_lag(D["payroll"], 3), 6) / 2.0
    f["payroll_amt_ratio_3m"] = _ratio(_rsum(D["payroll"], 3), prior_pay, 100.0, 0.0, 2.0)
    exp_n = np.round(_rsum(_lag(payroll_n, 3), 6) / 2.0)
    f["payroll_missed_cycles_3m"] = np.where(exp_n > 0, np.clip(exp_n - _rsum(payroll_n, 3), 0, 6), np.nan)
    f["inflow_ratio_3m"] = _ratio(_rsum(income_like, 3), _rsum(_lag(income_like, 3), 6) / 2.0, 100.0, 0.0, 2.0)
    f["transfer_in_share_3m"] = _ratio(_rsum(D["transfer_in_external"], 3), _rsum(inflow_all, 3), 1.0, 0.0, 1.0)

    buffer_days = np.clip(eom_bal / np.maximum(out3 / 90.0, 1.0), -30.0, 365.0)
    f["buffer_days"] = buffer_days
    f["buffer_days_change_6m"] = buffer_days - _lag(buffer_days, 6)
    f["min_balance_ratio_3m"] = np.clip(_rmin(min_bal, 3) / np.maximum(out3 / 3.0, 30.0), -2.0, 6.0)
    f["od_count_3m"] = _rsum(od_n, 3)
    f["od_months_6m"] = _rsum(od_n > 0, 6)
    f["neg_days_3m"] = _rsum(neg_days, 3)
    f["stl_months_6m"] = _rsum(D["shortterm_loan_proceeds"] > 0, 6)
    ext3 = _rsum(-D["loan_payment_external"], 3)
    f["ext_debt_pay_ratio_3m"] = _ratio(ext3, _rsum(income_like, 3), 100.0, 0.0, 1.5)
    ext_prior = _rsum(_lag(-D["loan_payment_external"], 3), 6) / 2.0
    # no payments before: "no change" if there are still none, the cap if they have just started
    f["ext_debt_pay_change_3m"] = np.where(
        ext_prior >= 50.0, _ratio(ext3, ext_prior, 50.0, 0.0, 3.0), np.where(ext3 >= 50.0, 3.0, 1.0)
    )
    f["legal_services_3m"] = (_rsum(-D["legal_services"], 3) > 0).astype(np.float64)
    for k in list(f):
        f[k] = np.where(has_dda[:, None], f[k], np.nan).astype(np.float32)

    # ------------------------------------------------------------- card features
    disc = -D["discretionary_debit"] + C["purchase_discretionary"]
    f["disc_spend_ratio_3m"] = _ratio(_rsum(disc, 3), _rsum(_lag(disc, 3), 6) / 2.0, 50.0, 0.0, 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        util = np.clip(bal / limit, 0.0, 1.5)
    f["utilization"] = util
    f["util_change_3m"] = util - _lag(util, 3)
    f["util_change_6m"] = util - _lag(util, 6)
    due_bal, due_min = _lag(bal, 1), _lag(min_due, 1)
    pr3 = _ratio(_rsum(pay, 3), _rsum(due_bal, 3), 1.0, 0.0, 1.2)
    f["pay_ratio_3m"] = np.where(np.isnan(pr3), 1.0, pr3)
    ptm = _ratio(_rsum(pay, 3), _rsum(due_min, 3), 1.0, 0.0, 20.0)
    f["pay_to_min_3m"] = np.where(np.isnan(ptm), 20.0, ptm)
    only_min = (due_min > 0) & (pay <= 1.1 * due_min) & (due_bal > 2 * due_min)
    f["min_pay_months_6m"] = _rsum(only_min, 6)
    pr_prior = _ratio(_rsum(_lag(pay, 3), 6), _rsum(_lag(due_bal, 3), 6), 1.0, 0.0, 1.2)
    f["pay_ratio_drift"] = f["pay_ratio_3m"] - np.where(np.isnan(pr_prior), 1.0, pr_prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["cash_adv_to_limit_3m"] = np.clip(_rsum(C["cash_advance"], 3) / limit, 0.0, 1.5)
    f["cash_adv_months_6m"] = _rsum(C["cash_advance"] > 0, 6)
    f["late_fee_count_6m"] = _rsum(Cn["fee_late"], 6)
    f["card_spend_ratio_3m"] = _ratio(_rsum(purch, 3), _rsum(_lag(purch, 3), 6) / 2.0, 50.0, 0.0, 3.0)
    f["max_cpd_12m"] = _rmax(np.nan_to_num(cpd), 12)
    f["statement_balance"] = bal
    f["open_to_buy"] = np.maximum(limit - bal, 0.0)

    # ------------------------------------------------------------- assemble
    eligible = is_open & (cpd == 0)
    eligible[:, :FIRST_SNAPSHOT] = False
    ii, tt = np.nonzero(eligible)
    out = {
        "customer_id": panel.customer_ids[ii].astype(np.int32),
        "snapshot_month": tt.astype(np.int16),
        "snapshot_date": panel.month_end_date(tt),
        "credit_limit": limit[ii, tt],
    }
    for feat in FEATURES + VALUE_EXTRAS:
        if feat.status == "prohibited":
            continue
        # Fixed precision: rolling sums carry float noise (1e-16) that would otherwise
        # become spurious bin edges, and a bank's own pipeline will not reproduce it.
        dollars = feat.name in ("statement_balance", "open_to_buy")
        col = f[feat.name][ii, tt].astype(np.float64)
        out[feat.name] = np.round(col, 2) if dollars else np.round(col, FEATURE_DECIMALS).astype(np.float32)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- labels
def build_labels(
    snapshots: pd.DataFrame,
    panel: Panel,
    statements: pd.DataFrame,
    outcomes: pd.DataFrame,
    interventions: pd.DataFrame | None = None,
    horizon: int = HORIZON,
    value_horizon: int = 18,
) -> pd.DataFrame:
    """Attach outcomes.  Uses future data by definition; never used as a feature."""
    N, T = len(panel.customer_ids), panel.n_months
    si = np.searchsorted(panel.customer_ids, statements["customer_id"].to_numpy())
    sm = panel.month_index(statements["statement_date"])
    cpd = np.zeros((N, T))
    cpd[si, sm] = statements["cycles_past_due"].to_numpy()
    loss = np.zeros((N, T))
    loss[si, sm] = statements["credit_loss"].to_numpy()
    rev = np.zeros((N, T))
    rev[si, sm] = statements["revenue"].to_numpy()

    oi = np.searchsorted(panel.customer_ids, outcomes["customer_id"].to_numpy())
    bk_m = np.full(N, -1)
    has = outcomes["bankruptcy_filing_date"].notna().to_numpy()
    bk_m[oi[has]] = panel.month_index(outcomes["bankruptcy_filing_date"][has])

    i = np.searchsorted(panel.customer_ids, snapshots["customer_id"].to_numpy())
    t = snapshots["snapshot_month"].to_numpy().astype(np.int64)

    def fwd(a, h, fn):
        """Aggregate a over months t+1 .. t+h."""
        c = np.concatenate([np.zeros((N, 1)), np.cumsum(a, axis=1)], axis=1)
        hi = np.minimum(t + h, T - 1)
        return fn(c[i, hi + 1] - c[i, t + 1])

    out = snapshots[["customer_id", "snapshot_month"]].copy()
    out["observed_6m"] = t + horizon <= T - 1
    out["y_dlq"] = fwd((cpd >= 2).astype(np.float64), horizon, lambda x: x > 0).astype(np.int8)
    out["y_bk"] = ((bk_m[i] > t) & (bk_m[i] <= t + horizon)).astype(np.int8)
    out["y_first_miss"] = fwd((cpd >= 1).astype(np.float64), horizon, lambda x: x > 0).astype(np.int8)
    out["loss_12m"] = fwd(loss, 12, lambda x: x)
    out["observed_12m"] = t + 12 <= T - 1
    out["loss_h"] = fwd(loss, value_horizon, lambda x: x)
    out["revenue_h"] = fwd(rev, value_horizon, lambda x: x)
    out["observed_h"] = t + value_horizon <= T - 1

    # randomized pilot: arm, and the weights that keep risk models on untreated outcomes
    out["arm"] = "not_enrolled"
    out["decision_month"] = -1
    out["action_cost"] = 0.0
    out["weight"] = 1.0
    out["propensity"] = 1.0
    out["is_decision_row"] = False
    if interventions is not None and len(interventions):
        dm = np.full(N, -1)
        arm = np.full(N, -1)
        cost = np.zeros(N)
        prop = np.ones(N)
        ii = np.searchsorted(panel.customer_ids, interventions["customer_id"].to_numpy())
        dm[ii] = panel.month_index(interventions["decision_date"])
        arm[ii] = pd.Categorical(interventions["arm"].astype(str), categories=ARMS).codes
        cost[ii] = interventions["action_cost"].to_numpy()
        prop[ii] = interventions["propensity"].to_numpy()
        a, d = arm[i], dm[i]
        out["arm"] = np.where(a >= 0, np.array(ARMS + ["not_enrolled"])[a], "not_enrolled")
        out["decision_month"] = d
        out["action_cost"] = cost[i]
        out["propensity"] = prop[i]
        touched = (a >= 0) & (t + horizon >= d)      # label window reaches the decision
        # treated rows are dropped (weight 0); control rows stand in for their whole stratum
        out["weight"] = np.where(touched, np.where(a == 0, 1.0 / prop[i], 0.0), 1.0)
        out["is_decision_row"] = (a >= 0) & (t == d)
    out["value_h"] = out["revenue_h"] - out["loss_h"] - out["action_cost"]
    return out


def assign_split(snapshot_month: np.ndarray, n_months: int, horizon: int = HORIZON) -> np.ndarray:
    """Out-of-time split with an embargo equal to the outcome horizon.

    train -> valid (early stopping, calibration, thresholds) -> embargo -> test.
    The embargo is the time it takes validation labels to mature, so the test
    window starts where a model frozen on those labels could first be used.
    """
    last = n_months - 1 - horizon
    test_len = 9
    test_start = last - test_len + 1
    valid_end = test_start - horizon - 1
    valid_start = valid_end - 5
    s = np.full(len(snapshot_month), "unlabelled", dtype=object)
    s[snapshot_month < valid_start] = "train"
    s[(snapshot_month >= valid_start) & (snapshot_month <= valid_end)] = "valid"
    s[(snapshot_month > valid_end) & (snapshot_month < test_start)] = "embargo"
    s[(snapshot_month >= test_start) & (snapshot_month <= last)] = "test"
    return s
