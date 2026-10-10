"""Feature registry: the single list of what the models may see.

Every feature carries
  * the direction in which it is allowed to move risk (monotone constraint),
  * a compliance status, and
  * the adverse-action statement used when it is a principal reason.

status
  allowed     used by default
  restricted  computed, but excluded from production models unless a bank's
              compliance function signs off (see docs/governance/feature-policy.md)
  prohibited  never computed and never accepted in a model bundle
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Feature:
    name: str
    group: str             # one adverse-action reason is reported per group
    source: str            # deposit | card
    monotone: int          # +1 risk rises with the value, -1 falls, 0 unconstrained
    reason_code: str
    reason_text: str
    description: str
    status: str = "allowed"
    note: str = ""

    @property
    def missing_text(self) -> str:
        """Reason statement when the value is absent and absence itself raises the score."""
        if self.group == "payroll":
            return "No regular payroll deposits on record with us"
        if self.group == "income":
            return "No regular income deposits on record with us"
        if self.source == "deposit":
            return "No deposit account history with us"
        return "Insufficient recent card activity to evaluate"


FEATURES: list[Feature] = [
    # ------------------------------------------------------------------ payroll gap
    Feature("payroll_gap_days", "payroll", "deposit", +1, "D01",
            "Length of time since the last regular payroll deposit",
            "Days since the most recent payroll credit beyond a normal monthly cycle: max(days - 31, 0). Zero for anyone paid within the last 31 days, so it does not move with month length. "
            "Missing when there is no payroll relationship or the last payroll is more than 120 days old."),
    Feature("payroll_amt_ratio_3m", "payroll", "deposit", -1, "D02",
            "Decrease in regular payroll deposits",
            "Payroll credited in the last 3 months divided by the 3-month average of the 6 months before that."),
    Feature("payroll_missed_cycles_3m", "payroll", "deposit", +1, "D03",
            "Expected payroll deposits not received",
            "Expected payroll count (from the prior 6 months) minus payroll credits received in the last 3 months."),
    Feature("inflow_ratio_3m", "income", "deposit", -1, "D04",
            "Decrease in total income deposited to the account",
            "Income-like credits (payroll, benefits, other income) in the last 3 months versus the 6 months before. Uses the amount of income from all sources and never which source it came from."),
    Feature("transfer_in_share_3m", "income", "deposit", 0, "D05",
            "Share of deposits arriving by transfer from other accounts",
            "External transfers in divided by all credits over 3 months."),
    # ------------------------------------------------------------------ balance buffer
    Feature("buffer_days", "buffer", "deposit", -1, "D06",
            "Deposit balance low relative to monthly spending",
            "Month-end balance divided by average daily outflow over the last 3 months."),
    Feature("buffer_days_change_6m", "buffer", "deposit", -1, "D07",
            "Decline in deposit balance relative to spending",
            "Change in buffer_days compared with 6 months earlier."),
    Feature("min_balance_ratio_3m", "buffer", "deposit", -1, "D08",
            "Low minimum deposit balance",
            "Lowest end-of-day balance in the last 3 months divided by average monthly outflow."),
    # ------------------------------------------------------------------ overdrafts
    Feature("od_count_3m", "overdraft", "deposit", +1, "D09",
            "Number of recent overdrafts",
            "Overdraft fees assessed in the last 3 months."),
    Feature("od_months_6m", "overdraft", "deposit", +1, "D10",
            "Frequency of overdrafts",
            "Months with at least one overdraft in the last 6."),
    Feature("neg_days_3m", "overdraft", "deposit", +1, "D11",
            "Number of days with a negative deposit balance",
            "Days ending with a negative balance in the last 3 months."),
    # ------------------------------------------------------------------ other debt
    Feature("stl_months_6m", "other_debt", "deposit", +1, "D12",
            "Recent short-term loan activity",
            "Months with proceeds from a short-term lender in the last 6."),
    Feature("ext_debt_pay_ratio_3m", "other_debt", "deposit", +1, "D13",
            "Payments to other lenders high relative to income deposited",
            "Payments to other lenders divided by income-like credits over 3 months."),
    Feature("ext_debt_pay_change_3m", "other_debt", "deposit", 0, "D14",
            "Change in payments to other lenders",
            "Payments to other lenders in the last 3 months versus the 6 months before."),
    Feature("disc_spend_ratio_3m", "spending", "card", 0, "C01",
            "Change in discretionary spending",
            "Discretionary spend (card and debit) in the last 3 months versus the 6 months before."),
    # ------------------------------------------------------------------ card behaviour
    Feature("utilization", "utilization", "card", +1, "C02",
            "Balance high relative to credit limit",
            "Statement balance divided by credit limit."),
    Feature("util_change_3m", "utilization_trend", "card", +1, "C03",
            "Recent increase in balance relative to credit limit",
            "Utilization now minus utilization 3 months ago."),
    Feature("util_change_6m", "utilization_trend", "card", +1, "C04",
            "Increase in balance relative to credit limit over six months",
            "Utilization now minus utilization 6 months ago."),
    Feature("pay_ratio_3m", "payments", "card", -1, "C05",
            "Payments low relative to balance",
            "Payments over 3 months divided by the statement balances they were due on."),
    Feature("pay_to_min_3m", "payments", "card", -1, "C06",
            "Payments close to the minimum amount due",
            "Payments over 3 months divided by minimum payments due."),
    Feature("min_pay_months_6m", "payments", "card", +1, "C07",
            "Number of months only the minimum payment was made",
            "Months in the last 6 with a payment at or within 10% of the minimum while carrying a balance."),
    Feature("pay_ratio_drift", "payment_trend", "card", -1, "C08",
            "Decrease in payment amounts",
            "Payment ratio over the last 3 months minus the payment ratio over the 6 months before."),
    Feature("cash_adv_to_limit_3m", "cash_advance", "card", +1, "C09",
            "Amount of recent cash advances",
            "Cash advances in the last 3 months divided by credit limit."),
    Feature("cash_adv_months_6m", "cash_advance", "card", +1, "C10",
            "Frequency of recent cash advances",
            "Months with a cash advance in the last 6."),
    Feature("late_fee_count_6m", "late_payments", "card", +1, "C11",
            "Recent late payments",
            "Late fees assessed in the last 6 months."),
    Feature("card_spend_ratio_3m", "spending", "card", 0, "C12",
            "Change in card spending",
            "Card purchases in the last 3 months versus the 6 months before."),
    Feature("max_cpd_12m", "delinquency_history", "card", +1, "C13",
            "Past delinquency on this account",
            "Highest number of cycles past due in the last 12 months (the account is current at the snapshot)."),
    # ------------------------------------------------------------------ restricted
    Feature("legal_services_3m", "restricted", "deposit", +1, "R01",
            "Recent payments to certain service providers",
            "Any debit to MCC 8111 (legal services) in the last 3 months.",
            status="restricted",
            note="Merchant-category behaviour. Predictive of bankruptcy in the simulator, but scoring customers on where they spend has drawn regulatory action and reads badly in an adverse-action notice. Excluded by default; its incremental lift is reported so a bank can decide."),
    # ------------------------------------------------------------------ prohibited
    Feature("gov_benefit_flag_3m", "prohibited", "deposit", +1, "X01",
            "(never used)",
            "Whether any deposit came from a government benefit programme.",
            status="prohibited",
            note="Receipt of public assistance income is a prohibited basis under ECOA / Regulation B (12 CFR 1002.2(z)). The amount of such income is counted inside inflow_ratio_3m; the fact of receiving it is never a feature."),
]

# Dollar quantities used only by the intervention value models, never by the risk scores.
VALUE_EXTRAS: list[Feature] = [
    Feature("statement_balance", "exposure", "card", 0, "V01", "", "Statement balance in dollars."),
    Feature("open_to_buy", "exposure", "card", 0, "V02", "", "Credit limit minus statement balance in dollars."),
]

BY_NAME = {f.name: f for f in FEATURES + VALUE_EXTRAS}


def model_features(include_restricted: bool = False) -> list[str]:
    ok = {"allowed", "restricted"} if include_restricted else {"allowed"}
    return [f.name for f in FEATURES if f.status in ok]


def value_features() -> list[str]:
    return model_features() + [f.name for f in VALUE_EXTRAS]


def assert_permitted(names: list[str], allow_restricted: bool = False) -> None:
    for n in names:
        f = BY_NAME.get(n)
        if f is None:
            raise ValueError(f"feature {n!r} is not in the registry")
        if f.status == "prohibited":
            raise ValueError(f"feature {n!r} is prohibited and cannot be used in a model")
        if f.status == "restricted" and not allow_restricted:
            raise ValueError(f"feature {n!r} is restricted; pass allow_restricted with documented sign-off")


def registry_json(names: list[str] | None = None) -> list[dict]:
    feats = FEATURES + VALUE_EXTRAS if names is None else [BY_NAME[n] for n in names]
    return [{**asdict(f), "missing_text": f.missing_text} for f in feats]
