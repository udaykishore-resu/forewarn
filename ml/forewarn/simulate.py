"""Synthetic retail bank.

Generates the four tables a bank already owns for its own customers
(deposit transactions, card transactions, card statements, account outcomes)
plus a randomized intervention pilot.  Nothing here is real customer data.

The simulator is mechanistic on purpose: distress is not sampled as a label and
then decorated with features.  Customers have income, bills, a cash balance and
a card, and they run a liquidity waterfall each month.  A job loss, an expense
shock or debt drift then *causes* the observable sequence (payroll gap ->
shrinking buffer -> spend moves to the card -> minimum payments -> overdrafts ->
missed payment), with lead times that depend on each customer's buffer.

Two things follow, and both matter when reading any metric produced from it:

  * the signals the models find are the signals this file puts in, and
  * effect sizes (lift, lead time, intervention benefit) are properties of
    these parameters, not evidence about a real portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DDA_CATEGORIES = [
    "payroll",                  # 0  ACH credit identified as employer payroll
    "gov_benefit",              # 1  government benefit credit (prohibited as a direct feature)
    "other_inflow",             # 2  irregular income: gig payouts, P2P, cash deposits
    "transfer_in_external",     # 3  transfer from an account at another institution
    "shortterm_loan_proceeds",  # 4  credit from a short-term / payday / BNPL-cash lender
    "cash_advance_deposit",     # 5  cash deposit that follows a card cash advance
    "housing",                  # 6  rent or mortgage
    "essential_debit",          # 7  groceries, utilities, fuel, insurance
    "discretionary_debit",      # 8  dining, travel, entertainment, retail
    "loan_payment_external",    # 9  payment to another lender
    "loan_payment_shortterm",   # 10 repayment to a short-term lender
    "card_payment_internal",    # 11 payment to this bank's own card
    "transfer_out_external",    # 12 transfer to an account elsewhere (savings sweep)
    "large_one_off",            # 13 single large debit
    "legal_services",           # 14 MCC 8111 (restricted feature, off by default)
    "fee_overdraft",            # 15 overdraft fee assessed by the bank
]
CARD_CATEGORIES = [
    "purchase_essential",       # 0
    "purchase_discretionary",   # 1
    "cash_advance",             # 2
    "fee_cash_advance",         # 3
    "payment",                  # 4 (negative amount)
    "interest",                 # 5
    "fee_late",                 # 6
]
ARMS = ["none", "due_date_change", "hardship_plan", "limit_decrease", "outreach_call"]
STATUS = ["open", "charged_off", "bankrupt", "closed"]


@dataclass
class SimConfig:
    n_customers: int = 40_000
    n_months: int = 54
    seed: int = 7
    start: str = "2022-01"            # calendar month of month index 0 (synthetic)
    burn_in: int = 24                 # months simulated and discarded so month 0 is a seasoned book
    # population mix
    share_has_dda: float = 0.80
    share_primary_given_dda: float = 0.85
    share_gig: float = 0.10
    share_transactor: float = 0.40
    share_drifter: float = 0.045
    # monthly hazards
    h_job_loss: float = 0.0036
    h_hours_cut: float = 0.0035
    h_job_change: float = 0.0080
    h_reemploy: float = 0.14
    h_hours_restore: float = 0.11
    h_expense_shock: float = 0.0045
    h_bank_switch: float = 0.0020
    h_attrition: float = 0.0030
    h_forget: float = 0.0040
    downturn_start: int = 42          # macro shift inside the out-of-time window
    downturn_multiplier: float = 1.5
    # card terms
    late_fee: float = 29.0
    overdraft_fee: float = 34.0
    min_due_floor: float = 35.0
    chargeoff_cycles: int = 6
    lgd: float = 0.90
    interchange: float = 0.015
    # randomized intervention pilot
    pilot_start: int = 12
    pilot_end: int = 36               # exclusive; leaves 18 months of outcome
    pilot_enroll_prob: float = 0.50
    hardship_months: int = 6
    hardship_min_due_factor: float = 0.0     # payment holiday, interest waived
    outreach_months: int = 9
    outreach_engage_prob: float = 0.40
    cost_due_date_change: float = 5.0
    cost_hardship_plan: float = 10.0
    cost_limit_decrease: float = 2.0
    cost_outreach_call: float = 15.0


class _Ledger:
    """Columnar accumulator; one append per event slot per month."""

    def __init__(self) -> None:
        self.cust: list[np.ndarray] = []
        self.month: list[np.ndarray] = []
        self.day: list[np.ndarray] = []
        self.amount: list[np.ndarray] = []
        self.cat: list[np.ndarray] = []
        self.seq: list[np.ndarray] = []

    def add(self, month: int, day, amount, cat: int, mask=None, seq: int = 0) -> None:
        amount = np.round(np.asarray(amount, dtype=np.float64), 2)
        m = amount != 0
        if mask is not None:
            m &= mask
        if not m.any():
            return
        n = int(m.sum())
        self.cust.append(np.flatnonzero(m).astype(np.int32))
        self.month.append(np.full(n, month, np.int16))
        self.day.append(np.broadcast_to(np.asarray(day), amount.shape)[m].astype(np.int8))
        self.amount.append(amount[m])
        self.cat.append(np.full(n, cat, np.int8))
        self.seq.append(np.full(n, seq, np.int8))

    def frame(self, start: str, categories: list[str]) -> pd.DataFrame:
        cust = np.concatenate(self.cust)
        month = np.concatenate(self.month)
        day = np.concatenate(self.day)
        seq = np.concatenate(self.seq)
        order = np.lexsort((seq, day, month, cust))
        base = np.datetime64(start, "M")
        dates = (base + month[order].astype("timedelta64[M]")).astype("datetime64[D]") + (
            day[order].astype(np.int64) - 1
        ).astype("timedelta64[D]")
        return pd.DataFrame(
            {
                "customer_id": cust[order],
                "posted_date": dates.astype("datetime64[s]"),
                "amount": np.concatenate(self.amount)[order],
                "category": pd.Categorical.from_codes(np.concatenate(self.cat)[order], categories),
            }
        )


NONE = -999   # "never happened" for month-valued truth columns


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ceil100(x):
    return np.ceil(x / 100.0) * 100.0


def _floor100(x):
    return np.floor(np.maximum(x, 0) / 100.0) * 100.0


def simulate(cfg: SimConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_customers, cfg.n_months
    U = rng.random

    # ------------------------------------------------------------------ static traits
    # Protected attributes exist only to test the models.  group_a is correlated
    # with income (so disparities can arise through legitimate financial factors);
    # group_b is independent of everything and works as a negative control.
    group_a = U(N) < 0.30
    group_b = U(N) < 0.50
    income_base = np.exp(rng.normal(np.log(4300.0), 0.45, N)) * np.where(group_a, 0.86, 1.0)
    income_base = np.clip(income_base, 1400.0, 25_000.0)

    has_dda = U(N) < cfg.share_has_dda
    primary = has_dda & (U(N) < cfg.share_primary_given_dda)
    gig = U(N) < cfg.share_gig
    freq = rng.choice(3, N, p=[0.50, 0.30, 0.20])       # 0 biweekly, 1 semimonthly, 2 monthly
    p1 = np.where(freq == 0, rng.integers(1, 14, N), np.where(freq == 1, 15, rng.choice([1, 15, 25, 28], N)))
    p2 = np.where(freq == 0, p1 + 14, np.where(freq == 1, 28, 0))

    transactor = U(N) < cfg.share_transactor
    drifter = U(N) < cfg.share_drifter
    drifter0 = drifter.copy()
    drift_start = rng.integers(-cfg.burn_in - 24, T, N)
    drift_rate = rng.uniform(0.20, 0.50, N)             # monthly overspend as share of income
    ca_prone = U(N) < 0.50
    st_prone = U(N) < 0.20
    card_first = U(N) < 0.60                            # protects the card before other debts
    sloppy = rng.beta(0.8, 2.5, N)                      # P(skip a cycle when cash is short on the due day)
    pay_lead = rng.integers(0, 4, N)

    housing = income_base * rng.uniform(0.22, 0.36, N)
    essentials = income_base * rng.uniform(0.22, 0.34, N)
    surplus = np.clip(rng.normal(0.06, 0.06, N), -0.04, 0.25)
    ext_debt = income_base * 12 * np.minimum(np.exp(rng.normal(np.log(0.08), 1.0, N)), 0.50)
    ext_debt[U(N) < 0.25] = 0.0
    # drift that began before the simulated period arrives as debt already accumulated
    pre = np.where(drifter, np.maximum(-cfg.burn_in - drift_start, 0), 0) * drift_rate * income_base
    ext_debt += 0.55 * pre

    apr = rng.uniform(0.20, 0.29, N)
    limit = np.clip(np.round(income_base * rng.uniform(0.8, 2.5, N) / 100.0) * 100.0, 500.0, 30_000.0)
    due_day = rng.integers(3, 28, N)
    c_e = np.where(transactor, rng.uniform(0.5, 0.8, N), rng.uniform(0.2, 0.5, N))
    c_d = np.where(transactor, rng.uniform(0.6, 0.9, N), rng.uniform(0.3, 0.7, N))
    rho = np.where(drifter, rng.uniform(0.60, 0.85, N), rng.uniform(0.92, 1.05, N))
    paydown = rng.uniform(0.0, 0.03, N)

    disc_base = np.maximum(
        income_base * (1 - surplus) - housing - essentials - 0.03 * ext_debt, 0.04 * income_base
    )
    monthly_out = housing + essentials + disc_base + 0.03 * ext_debt
    card_bal = np.where(
        transactor,
        essentials * c_e + disc_base * c_d,
        limit * rng.beta(2.0, 4.0, N),
    )
    card_bal = np.minimum(card_bal + 0.45 * pre, limit * 0.95)
    savings = income_base * np.exp(rng.normal(np.log(0.6), 1.2, N))
    cash = monthly_out * np.exp(rng.normal(np.log(0.8), 0.7, N))
    sec_buf = rng.uniform(100.0, 1500.0, N)
    sec_spend = sec_buf * rng.uniform(0.10, 0.40, N)
    opening_balance = np.full(N, np.nan)

    # ------------------------------------------------------------------ dynamic state
    income_cur = income_base.copy()
    active = np.ones(N, bool)
    status = np.zeros(N, np.int8)
    emp = np.zeros(N, np.int8)                      # 0 employed, 1 reduced hours, 2 unemployed
    unemp_months = np.zeros(N, np.int16)
    benefit_left = np.zeros(N, np.int16)
    cpd = np.zeros(N, np.int8)                      # cycles past due
    min_due_prev = np.where(card_bal > 0, np.minimum(card_bal, np.maximum(cfg.min_due_floor, 0.01 * card_bal)), 0.0)
    base_min_prev = min_due_prev.copy()
    interest_prev = np.zeros(N)
    purch_prev = essentials * c_e + disc_base * c_d
    disc_mult = np.ones(N)
    stl_due = np.zeros(N)
    prefile = np.zeros(N, np.int8)                  # months until filing; 0 = not in pre-filing
    bk_style = np.zeros(N, np.int8)                 # 1 run-up, 2 quiet, 3 stops paying
    bk_stop_ext = np.zeros(N, bool)
    bk_legal_month = np.full(N, NONE, np.int16)
    ext_delinq = np.zeros(N, np.int16)
    hardship_left = np.zeros(N, np.int16)
    outreach_left = np.zeros(N, np.int16)
    limit_cut_left = np.zeros(N, np.int16)
    arm = np.full(N, -1, np.int8)
    arm_month = np.full(N, -1, np.int16)
    ever_payroll = np.zeros(N, bool)
    mo_since_payroll = np.zeros(N, np.int16)

    # rolling signals for the legacy pilot trigger
    od_hist = [np.zeros(N, np.int8) for _ in range(3)]
    minpay_hist = [np.zeros(N, bool) for _ in range(6)]
    ca_hist = [np.zeros(N, bool) for _ in range(6)]
    util_hist = [card_bal / limit for _ in range(4)]

    # ------------------------------------------------------------------ recorders
    f32 = lambda: np.zeros((T, N), np.float32)  # noqa: E731
    rec_bal, rec_limit, rec_min, rec_pay, rec_purch = f32(), f32(), f32(), f32(), f32()
    rec_rev, rec_loss, rec_pen = f32(), f32(), f32()
    rec_cpd = np.zeros((T, N), np.int8)
    rec_status = np.zeros((T, N), np.int8)
    rec_active = np.zeros((T, N), bool)
    rec_due = np.zeros((T, N), np.int8)
    rec_apr = f32()
    truth_job_loss = np.full(N, NONE, np.int16)
    truth_shock = np.full(N, NONE, np.int16)
    truth_prefile = np.full(N, NONE, np.int16)
    truth_switch = np.full(N, NONE, np.int16)
    close_month = np.full(N, -1, np.int16)
    chargeoff_month = np.full(N, -1, np.int16)
    bankruptcy_month = np.full(N, -1, np.int16)
    loss_amount = np.zeros(N)

    dda, card = _Ledger(), _Ledger()
    season = np.array([0.92, 0.95, 1.0, 1.0, 1.02, 1.05, 1.10, 1.05, 1.0, 1.0, 1.10, 1.35])
    start_month_of_year = int(cfg.start.split("-")[1]) - 1

    for t in range(-cfg.burn_in, T):
        rec = t >= 0                     # burn-in months are simulated but not recorded
        if t == 0:
            opening_balance = np.where(primary, cash, np.where(has_dda, sec_buf, np.nan))
            has_dda_at_start = has_dda & active
        macro = cfg.downturn_multiplier if t >= cfg.downturn_start else 1.0
        a = active

        # ---------------------------------------------------------- A. state transitions
        lose = a & (emp != 2) & (U(N) < cfg.h_job_loss * macro)
        cut = a & (emp == 0) & ~lose & (U(N) < cfg.h_hours_cut * macro)
        back = a & (emp == 2) & ~lose & (U(N) < cfg.h_reemploy / np.sqrt(macro))
        restore = a & (emp == 1) & ~cut & (U(N) < cfg.h_hours_restore)
        change = a & (emp == 0) & ~lose & ~cut & (U(N) < cfg.h_job_change)
        truth_job_loss[lose] = t
        emp[lose] = 2
        unemp_months[lose] = 0
        benefit_left[lose & ~gig] = 6
        emp[cut] = 1
        emp[back] = 0
        income_cur[back] = income_cur[back] * rng.uniform(0.80, 1.05, int(back.sum()))
        emp[restore] = 0
        income_cur[change] = income_cur[change] * rng.uniform(0.95, 1.25, int(change.sum()))
        newday = rng.integers(1, 14, N)
        p1 = np.where(change & (freq == 0), newday, p1)
        p2 = np.where(change & (freq == 0), newday + 14, p2)
        unemp_months[emp == 2] += 1

        switch = a & primary & (U(N) < cfg.h_bank_switch)
        truth_switch[switch] = t

        shock = a & (U(N) < cfg.h_expense_shock)
        shock_amt = np.where(shock, income_base * rng.uniform(0.8, 3.0, N), 0.0)
        truth_shock[shock] = t

        drifting = a & drifter & (t >= drift_start) & (prefile == 0)

        debt_ratio = (card_bal + ext_debt) / (income_base * 12)
        h_bk = (
            0.0003
            + 0.05 * _sigmoid((debt_ratio - 0.95) / 0.08)
            + 0.015 * ((unemp_months >= 4) & (emp == 2) & (debt_ratio > 0.25))
            + 0.010 * (ext_delinq >= 3)
        )
        enter = a & (prefile == 0) & (U(N) < h_bk)
        prefile[enter] = rng.integers(4, 9, int(enter.sum()))
        bk_style[enter] = rng.choice([1, 2, 3], int(enter.sum()), p=[0.45, 0.30, 0.25])
        bk_stop_ext[enter] = U(int(enter.sum())) < 0.60
        legal = enter & (U(N) < 0.45)
        bk_legal_month[legal] = t + np.maximum(prefile[legal] - rng.integers(1, 3, int(legal.sum())), 0)
        truth_prefile[enter] = t
        in_prefile = a & (prefile > 0)
        runup = in_prefile & (bk_style == 1)
        stop_pay = in_prefile & (bk_style == 3)

        engaged = a & (outreach_left > 0)

        # ---------------------------------------------------------- B. income
        scale = np.where(emp == 2, 0.0, np.where(emp == 1, 0.60, 1.0))
        per_check = np.where(freq == 2, income_cur, income_cur / 2.0) * scale
        pay1 = np.where(gig, 0.0, per_check)
        pay2 = np.where(gig | (freq == 2), 0.0, per_check)
        # the month the job ends the first cheque still arrives
        pay1 = np.where(lose & ~gig, np.where(freq == 2, income_cur * (U(N) < 0.5), income_cur / 2.0), pay1)
        pay1 = np.where(change & (freq != 2), 0.0, pay1)          # job change: one cheque skipped
        d_pay1 = np.where(change & (freq == 2), np.minimum(p1 + 10, 28), p1)
        d_pay2 = p2.copy()
        benefit = np.where((emp == 2) & (benefit_left > 0) & ~lose, 0.35 * income_cur, 0.0)
        benefit_left[(emp == 2) & ~lose & (benefit_left > 0)] -= 1
        gig_scale = np.where(emp == 2, 0.25, np.where(emp == 1, 0.6, 1.0))
        other1 = np.where(gig, income_cur / 2.0 * np.exp(rng.normal(0, 0.35, N)) * gig_scale, 0.0)
        other2 = np.where(gig, income_cur / 2.0 * np.exp(rng.normal(0, 0.35, N)) * gig_scale, 0.0)
        inc = pay1 + pay2 + benefit + other1 + other2
        got_payroll = (pay1 + pay2) > 0
        last_pay_day = np.where(pay2 > 0, d_pay2, np.where(pay1 > 0, d_pay1, 0))

        # ---------------------------------------------------------- C. planned spending
        moy = (start_month_of_year + t) % 12
        E = essentials * np.exp(rng.normal(0, 0.06, N))
        D = disc_base * disc_mult * season[moy] * np.exp(rng.normal(0, 0.15, N))
        D = np.where(engaged, D * 0.75, D)
        X = 0.03 * ext_debt
        X = np.where(in_prefile & bk_stop_ext, 0.0, X)
        drift_over = np.where(drifting, drift_rate * income_base, 0.0)

        frozen = hardship_left > 0       # hardship plan: no new purchases or advances
        otb = np.where(frozen, 0.0, np.maximum(limit - card_bal, 0.0))
        Pe = np.minimum(E * c_e, otb)
        Pd = np.minimum(D * c_d * np.where(runup, 1.6, 1.0), otb - Pe)
        drift_card = np.minimum(drift_over * 0.45, np.maximum(otb - Pe - Pd, 0.0))
        ext_debt += drift_over - drift_card
        Ed = E - Pe
        Dd = D - np.minimum(Pd, D)
        shock_cash = np.minimum(shock_amt, np.maximum(cash + savings * 0.5, 0.0))
        shock_card = np.minimum(shock_amt - shock_cash, np.maximum(otb - Pe - Pd - drift_card, 0.0))
        ext_debt += shock_amt - shock_cash - shock_card
        otb = np.maximum(otb - Pe - Pd - drift_card - shock_card, 0.0)

        # ---------------------------------------------------------- D. card payment intent
        eff_min = min_due_prev * np.where(hardship_left > 0, cfg.hardship_min_due_factor, 1.0)
        intend = np.where(
            transactor,
            card_bal,
            np.maximum(eff_min, interest_prev + purch_prev * rho + paydown * card_bal),
        )
        intend = np.minimum(intend, card_bal)
        intend = np.where(runup, np.minimum(eff_min, card_bal), intend)
        intend = np.where(stop_pay, 0.0, intend)
        stl_repay = stl_due.copy()
        stl_due[:] = 0.0

        # ---------------------------------------------------------- E. liquidity waterfall
        musts = housing + Ed + X + stl_repay + shock_cash
        short = np.maximum(musts + Dd + intend - (cash + inc), 0.0)
        draw = np.minimum(short, savings)
        savings -= draw
        short -= draw
        stressed = short > 1.0
        cutd = np.minimum(short, 0.7 * Dd)
        Dd -= cutd
        short -= cutd
        shift = np.minimum(np.minimum(short, otb), Ed + Dd)
        shift_d = np.minimum(shift, Dd)
        shift_e = shift - shift_d
        Dd -= shift_d
        Ed -= shift_e
        otb -= shift
        short -= shift
        red = np.minimum(short, np.maximum(intend - eff_min, 0.0))
        pay = intend - red
        short -= red
        adv = np.where(ca_prone & (short > 1.0) & ~engaged, np.minimum(_ceil100(short), _floor100(otb)), 0.0)
        adv += np.where(runup, _floor100(rng.uniform(0.10, 0.25, N) * np.maximum(otb - adv, 0.0)), 0.0)
        # occasional small convenience advances so the category is not a pure distress tell
        adv += np.where(ca_prone & a & (adv == 0) & (U(N) < 0.006), np.minimum(rng.integers(1, 5, N) * 100.0, _floor100(otb)), 0.0)
        otb -= adv
        short = np.maximum(short - adv, 0.0)
        stl = np.where(st_prone & (short > 1.0) & ~engaged, np.minimum(_ceil100(short), 1500.0), 0.0)
        stl += np.where(drifting & st_prone & (debt_ratio > 0.5) & (U(N) < 0.25), 500.0, 0.0)
        stl_due += 1.2 * stl
        short = np.maximum(short - stl, 0.0)
        skip_x1 = np.where(card_first, np.minimum(short, X), 0.0)
        short -= skip_x1
        skip_card = np.minimum(short, pay)
        pay -= skip_card
        short -= skip_card
        skip_x2 = np.minimum(short, X - skip_x1)
        short -= skip_x2
        Xp = X - skip_x1 - skip_x2
        ext_delinq = np.where(Xp < X - 1.0, ext_delinq + 1, 0)
        # whatever is still uncovered goes unpaid (bills first, then housing); the
        # bank tolerates only a small overdraft, so balances cannot run away
        proj_end = cash + inc + draw + adv + stl - (housing + Ed + Dd + Xp + stl_repay + shock_cash + pay)
        deficit = np.maximum(-300.0 * U(N) - proj_end, 0.0)
        cut2 = np.minimum(deficit, Dd)
        Dd -= cut2
        deficit -= cut2
        cut2 = np.minimum(deficit, Ed)
        Ed -= cut2
        deficit -= cut2
        housing_paid = housing - np.minimum(deficit, housing)
        disc_mult = np.where(stressed, np.maximum(0.40, disc_mult * 0.80), np.minimum(1.0, disc_mult * 1.10))

        # ---------------------------------------------------------- F. intramonth timing
        legal_fee = np.where(a & (bk_legal_month == t), rng.uniform(900.0, 1800.0, N), 0.0)
        off = (np.arange(N) % 5).astype(np.int64)
        K = 19
        days = np.zeros((N, K), np.int64)
        amts = np.zeros((N, K))
        slots = [
            (d_pay1, pay1), (d_pay2, pay2), (10, benefit), (8, other1), (22, other2),
            (2, draw), (4, stl), (5, adv),
            (1 + off, -housing_paid), (6 + off, -Ed / 2), (20 + off, -Ed / 2),
            (10 + off, -Dd / 2), (23 + off, -Dd / 2), (12 + off, -Xp),
            (3, -stl_repay), (9, -shock_cash), (11, -legal_fee), (27, 0.0), (0, 0.0),
        ]
        for k, (d, v) in enumerate(slots):
            days[:, k] = d
            amts[:, k] = v
        SWEEP, CARDPAY = 17, 18

        # Customers pay right after the last cheque that lands on or before the due
        # date.  If even the minimum is not in the account that day, they either pay
        # late after the next cheque or let the cycle slip.
        c1 = (pay1 > 0) & (d_pay1 <= due_day)
        c2 = (pay2 > 0) & (d_pay2 <= due_day)
        pay_day = np.where(c2, d_pay2, np.where(c1, d_pay1, np.clip(due_day - pay_lead, 1, 28)))
        on_hand = np.maximum(cash + (amts * (days <= pay_day[:, None])).sum(axis=1), 0.0)
        need = np.minimum(pay, eff_min)
        pay = np.where((on_hand < pay) & (on_hand >= need), on_hand, pay)
        later_pay = last_pay_day > pay_day
        tight = a & (pay > 0) & (on_hand < need) & later_pay
        skip_cycle = tight & (U(N) < sloppy)
        forget = a & (pay > 0) & ~tight & (U(N) < cfg.h_forget)
        pay = np.where(skip_cycle | forget, 0.0, pay)
        pay_late = tight & ~skip_cycle
        pay_day = np.where(pay_late, np.minimum(last_pay_day + 1, 28), pay_day)
        late_fee = np.where(a & (pay > 0) & (pay_day > due_day) & (eff_min > 0), cfg.late_fee, 0.0)
        days[:, CARDPAY] = pay_day
        amts[:, CARDPAY] = -pay
        amts[~a] = 0.0

        # customers push bills past payday when the account would otherwise go negative
        order = np.argsort(days, axis=1, kind="stable")
        run = cash[:, None] + np.cumsum(np.take_along_axis(amts, order, 1), axis=1)
        run_o = np.empty_like(run)
        np.put_along_axis(run_o, order, run, 1)
        for k in (8, 9, 10, 11, 12, 13):
            mv = (run_o[:, k] < 0) & (last_pay_day > days[:, k]) & (U(N) < 0.8)
            days[:, k] = np.where(mv, np.minimum(last_pay_day + 1, 28), days[:, k])
        order = np.argsort(days, axis=1, kind="stable")
        a_sorted = np.take_along_axis(amts, order, 1)
        run = cash[:, None] + np.cumsum(a_sorted, axis=1)
        n_od = ((run < -0.005) & (a_sorted < 0)).sum(axis=1)
        n_od = np.where(primary & a, np.minimum(n_od, 3), 0)
        od_fee = n_od * cfg.overdraft_fee
        cash_end = run[:, -1] - np.where(primary, od_fee, 0.0)

        monthly_out_now = housing + E + D + X
        sweep = np.where(a & ~stressed & (cash_end > 2.2 * monthly_out_now), 0.5 * (cash_end - 2.0 * monthly_out_now), 0.0)
        savings += sweep
        cash_end -= sweep
        amts[:, SWEEP] = -sweep

        # ---------------------------------------------------------- G. ledgers
        vis = primary & a
        if rec:
            dda_slot_cat = [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 13, 14, 12, 11]
            for k in range(K):
                dda.add(t, days[:, k], amts[:, k], dda_slot_cat[k], vis, seq=k)
            dda.add(t, 28, -od_fee, 15, vis, seq=K)
            if switch.any():
                # leaves most of the balance behind a transfer to the new primary bank
                move = cash_end - sec_buf
                dda.add(t, 28, -np.maximum(move, 0.0), 12, switch, seq=K + 1)
                dda.add(t, 28, np.maximum(-move, 0.0), 3, switch, seq=K + 1)
            sec = has_dda & ~primary & a
            dda.add(t, np.maximum(pay_day - 1, 1), pay + sec_spend, 3, sec, seq=0)
            dda.add(t, 12, -sec_spend, 8, sec, seq=1)
            dda.add(t, pay_day, -pay, 11, sec, seq=2)
        primary = primary & ~switch

        pe_total = Pe + shift_e + shock_card
        pd_total = Pd + shift_d + drift_card
        adv_fee = np.where(adv > 0, np.maximum(0.05 * adv, 10.0), 0.0)
        hard = hardship_left > 0
        revolve = pay < card_bal - 0.005
        interest = np.where(revolve & ~hard, apr / 12.0 * np.maximum(card_bal - pay, 0.0), 0.0)
        # a missed cycle carries a late fee too, and it must be on the ledger
        missed = a & (min_due_prev > 0.005) & (pay < eff_min - 0.01)
        cured = a & ~missed
        late_fee = np.where(missed, cfg.late_fee, late_fee)
        if rec:
            card.add(t, 14, pe_total, 0, a, seq=0)
            card.add(t, 18, pd_total, 1, a, seq=1)
            card.add(t, 5, adv, 2, a, seq=0)
            card.add(t, 5, adv_fee, 3, a, seq=1)
            card.add(t, pay_day, -pay, 4, a, seq=0)
            card.add(t, 28, interest, 5, a, seq=2)
            card.add(t, np.minimum(due_day + 1, 28), late_fee, 6, a, seq=3)

        # ---------------------------------------------------------- H. statement
        new_bal = np.where(a, card_bal - pay + pe_total + pd_total + adv + adv_fee + interest + late_fee, 0.0)
        new_bal = np.maximum(new_bal, 0.0)
        cpd = np.where(missed, cpd + 1, np.where(cured, 0, cpd)).astype(np.int8)
        base_min = np.maximum(cfg.min_due_floor, 0.01 * new_bal + interest + late_fee)
        min_due = np.minimum(new_bal, base_min * (1 + cpd))
        revenue = np.where(a, interest + adv_fee + cfg.interchange * (pe_total + pd_total), 0.0)
        penalty = np.where(a, late_fee, 0.0)

        minpay = a & (min_due_prev > 0.005) & (pay <= 1.1 * eff_min) & (card_bal > 2 * eff_min)
        util = np.where(limit > 0, new_bal / limit, 0.0)

        # ---------------------------------------------------------- I. outcomes
        prefile[in_prefile] -= 1
        file_bk = a & in_prefile & (prefile == 0)
        file_bk |= a & (cpd >= 2) & (U(N) < 0.03)
        co = a & ~file_bk & (cpd >= cfg.chargeoff_cycles)
        attr_p = cfg.h_attrition + np.where((limit_cut_left > 0) & (cpd == 0) & ~stressed, 0.04, 0.0)
        leave = a & ~file_bk & ~co & (cpd == 0) & (new_bal < 0.3 * limit) & (U(N) < attr_p)
        loss = np.where(file_bk | co, cfg.lgd * new_bal, 0.0)
        status = np.where(file_bk, 2, np.where(co, 1, np.where(leave, 3, status))).astype(np.int8)
        if rec:
            bankruptcy_month[file_bk] = t
            chargeoff_month[co] = t
            close_month[file_bk | co | leave] = t
            loss_amount += loss
            rec_bal[t], rec_limit[t], rec_min[t], rec_pay[t] = new_bal, limit, min_due, np.where(a, pay, 0.0)
            rec_purch[t], rec_rev[t], rec_loss[t] = np.where(a, pe_total + pd_total, 0.0), revenue, loss
            rec_pen[t] = penalty
            rec_cpd[t], rec_status[t], rec_active[t], rec_due[t], rec_apr[t] = cpd, status, a, due_day, apr

        # ---------------------------------------------------------- J. randomized pilot
        od_hist = od_hist[1:] + [n_od.astype(np.int8)]
        minpay_hist = minpay_hist[1:] + [minpay]
        ca_hist = ca_hist[1:] + [adv > 0]
        util_hist = util_hist[1:] + [util]
        ever_payroll |= got_payroll & vis
        mo_since_payroll = np.where(got_payroll, 0, mo_since_payroll + 1)
        if cfg.pilot_start <= t < cfg.pilot_end:
            trig = (
                ((sum(od_hist) >= 1) & vis)
                | (ever_payroll & vis & (mo_since_payroll >= 1))
                | ((util >= 0.5) & (sum(h.astype(np.int8) for h in minpay_hist) >= 2))
                | (sum(h.astype(np.int8) for h in ca_hist) >= 1)
                | (util - util_hist[0] >= 0.15)
            )
            enroll = a & ~file_bk & ~co & ~leave & (cpd == 0) & (arm < 0) & trig & (U(N) < cfg.pilot_enroll_prob)
            k = int(enroll.sum())
            if k:
                arm[enroll] = rng.integers(0, len(ARMS), k).astype(np.int8)
                arm_month[enroll] = t
                e1 = enroll & (arm == 1) & (last_pay_day > 0)
                due_day = np.where(e1, np.clip(last_pay_day + 2, 3, 27), due_day)
                pay_lead = np.where(e1, 0, pay_lead)
                hardship_left[enroll & (arm == 2)] = cfg.hardship_months + 1
                e3 = enroll & (arm == 3)
                limit = np.where(e3, np.minimum(limit, np.maximum(np.ceil(new_bal * 1.05 / 50.0) * 50.0, 300.0)), limit)
                limit_cut_left[e3] = 7
                e4 = enroll & (arm == 4) & (U(N) < cfg.outreach_engage_prob)
                outreach_left[e4] = cfg.outreach_months + 1
                stopdrift = e4 & drifter & (U(N) < 0.60)
                drifter = drifter & ~stopdrift
                exitbk = e4 & (prefile > 0) & (U(N) < 0.50)
                prefile[exitbk] = 0
        hardship_left = np.maximum(hardship_left - 1, 0)
        outreach_left = np.maximum(outreach_left - 1, 0)
        limit_cut_left = np.maximum(limit_cut_left - 1, 0)

        # ---------------------------------------------------------- K. roll forward
        active = a & ~file_bk & ~co & ~leave
        cash = np.where(a, cash_end, cash)
        card_bal = new_bal
        min_due_prev, base_min_prev = min_due, base_min
        interest_prev = interest
        purch_prev = pe_total + pd_total

    # ---------------------------------------------------------------------- frames
    cid = np.arange(N, dtype=np.int32)
    base = np.datetime64(cfg.start, "M")
    month_end = lambda m: ((base + np.asarray(m).astype("timedelta64[M]") + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D"))  # noqa: E731

    tt, cc = np.nonzero(rec_active)
    statements = pd.DataFrame(
        {
            "customer_id": cc.astype(np.int32),
            "statement_date": month_end(tt).astype("datetime64[s]"),
            "statement_balance": np.round(rec_bal[tt, cc].astype(np.float64), 2),
            "credit_limit": rec_limit[tt, cc].astype(np.float64),
            "min_due": np.round(rec_min[tt, cc].astype(np.float64), 2),
            "payments": np.round(rec_pay[tt, cc].astype(np.float64), 2),
            "purchases": np.round(rec_purch[tt, cc].astype(np.float64), 2),
            "apr": rec_apr[tt, cc].astype(np.float64),
            "due_day": rec_due[tt, cc],
            "cycles_past_due": rec_cpd[tt, cc],
            "status": pd.Categorical.from_codes(rec_status[tt, cc], STATUS),
            "revenue": np.round(rec_rev[tt, cc].astype(np.float64), 2),
            "penalty_fees": np.round(rec_pen[tt, cc].astype(np.float64), 2),
            "credit_loss": np.round(rec_loss[tt, cc].astype(np.float64), 2),
        }
    ).sort_values(["customer_id", "statement_date"], kind="stable").reset_index(drop=True)

    book = rec_active[0]                 # accounts open at month 0
    customers = pd.DataFrame(
        {
            "customer_id": cid[book],
            "has_dda": has_dda_at_start[book],
            "dda_opening_balance": np.round(opening_balance, 2)[book],
        }
    )
    protected = pd.DataFrame({"customer_id": cid[book], "group_a": group_a[book], "group_b": group_b[book]})
    def as_date(m):
        d = month_end(np.maximum(m, 0)).astype("datetime64[s]")
        return pd.Series(d).where(pd.Series(m) >= 0)

    outcomes = pd.DataFrame(
        {
            "customer_id": cid,
            "chargeoff_date": as_date(chargeoff_month),
            "bankruptcy_filing_date": as_date(bankruptcy_month),
            "close_date": as_date(close_month),
            "loss_amount": np.round(loss_amount, 2),
        }
    )[book].reset_index(drop=True)
    enrolled = arm >= 0
    costs = np.array([0.0, cfg.cost_due_date_change, cfg.cost_hardship_plan, cfg.cost_limit_decrease, cfg.cost_outreach_call])
    interventions = pd.DataFrame(
        {
            "customer_id": cid[enrolled],
            "decision_date": month_end(arm_month[enrolled]).astype("datetime64[s]"),
            "arm": pd.Categorical.from_codes(arm[enrolled], ARMS),
            "propensity": 1.0 / len(ARMS),
            "action_cost": costs[arm[enrolled]],
        }
    )
    truth = pd.DataFrame(
        {
            "customer_id": cid,
            "income_base": np.round(income_base, 2),
            "gig": gig,
            "transactor": transactor,
            "drifter": drifter0,
            "drift_start_month": np.where(drifter0, drift_start, -99),
            "start_month": cfg.start,
            "job_loss_month": truth_job_loss,
            "expense_shock_month": truth_shock,
            "prefile_month": truth_prefile,
            "bank_switch_month": truth_switch,
            "bk_style": bk_style,
            "sloppy": np.round(sloppy, 3),
        }
    )[book].reset_index(drop=True)
    return {
        "customers": customers,
        "protected_attributes": protected,
        "dda_transactions": dda.frame(cfg.start, DDA_CATEGORIES),
        "card_transactions": card.frame(cfg.start, CARD_CATEGORIES),
        "card_statements": statements,
        "outcomes": outcomes,
        "interventions": interventions,
        "sim_truth": truth,
        "sim_config": pd.DataFrame([asdict(cfg)]),
    }
