# Data contract

Forewarn needs four tables for scoring and two more for training. All of them describe the bank's own accounts. No bureau file, no data from another institution, no open-banking feed.

Dates are calendar dates. Amounts are US dollars. One row per event.

## Scoring inputs

### `customers`

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | Stable identifier, the join key everywhere |
| `has_dda` | boolean | Customer holds a deposit account at the bank |
| `dda_opening_balance` | decimal | Deposit balance on the day before the first transaction supplied |

### `dda_transactions`

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | |
| `posted_date` | date | |
| `amount` | decimal | Credits positive, debits negative |
| `category` | enum | One of the categories below |

| Category | Sign | Typical source in a core system |
|---|---|---|
| `payroll` | + | ACH credits identified as employer payroll |
| `gov_benefit` | + | Government benefit credits |
| `other_inflow` | + | Irregular income: gig payouts, P2P, cash deposits |
| `transfer_in_external` | + | Transfers from accounts at other institutions |
| `shortterm_loan_proceeds` | + | Credits from short-term or payday lenders |
| `cash_advance_deposit` | + | Cash deposits following a card cash advance |
| `housing` | − | Rent or mortgage |
| `essential_debit` | − | Groceries, utilities, fuel, insurance |
| `discretionary_debit` | − | Dining, travel, entertainment, retail |
| `loan_payment_external` | − | Payments to other lenders |
| `loan_payment_shortterm` | − | Repayments to short-term lenders |
| `card_payment_internal` | − | Payments to this bank's card |
| `transfer_out_external` | − | Transfers to accounts elsewhere |
| `large_one_off` | − | Single unusually large debit |
| `legal_services` | − | MCC 8111. Feeds a restricted feature that is off by default |
| `fee_overdraft` | − | Overdraft fees assessed |

`gov_benefit` amounts count toward total income deposited. Whether a customer receives benefits is never a feature (see `governance/feature-policy.md`).

The mapping from raw transactions to these categories is the bank's work and is the largest integration task. Categorisation errors are the first thing to check when a feature's stability index moves.

### `card_transactions`

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | |
| `posted_date` | date | |
| `amount` | decimal | Increases to the balance positive, payments negative |
| `category` | enum | `purchase_essential`, `purchase_discretionary`, `cash_advance`, `fee_cash_advance`, `payment`, `interest`, `fee_late` |

### `card_statements`

One row per account per cycle.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | |
| `statement_date` | date | Cycle end |
| `statement_balance` | decimal | |
| `credit_limit` | decimal | |
| `min_due` | decimal | Minimum payment due in the next cycle |
| `payments` | decimal | Payments received during this cycle |
| `purchases` | decimal | Purchases during this cycle |
| `cycles_past_due` | integer | 0 current, 1 = 30 days, 2 = 60 days, ... |
| `status` | enum | `open`, `charged_off`, `bankrupt`, `closed` |

The simulator also writes `apr`, `due_day`, `revenue`, `penalty_fees` and `credit_loss`. The last three are used only to build training labels.

## Training inputs

### `outcomes`

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | |
| `chargeoff_date` | date or null | |
| `bankruptcy_filing_date` | date or null | |
| `close_date` | date or null | |
| `loss_amount` | decimal | Credit loss booked |

### `interventions`

Needed only for the action-recommendation layer. One row per account enrolled in a randomized pilot.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | integer | |
| `decision_date` | date | Month end on which the arm was assigned |
| `arm` | enum | `none`, `due_date_change`, `hardship_plan`, `limit_decrease`, `outreach_call` |
| `propensity` | decimal | Probability of receiving that arm under the pilot design |
| `action_cost` | decimal | Operating cost of the action |

Assignment must be random, with known probabilities. Actions chosen by people cannot stand in for it.

### `protected_attributes`

Used by fair-lending tests only and joined by the validation step alone. Never a model input. A bank would supply whatever it lawfully holds or imputes for testing.

## What the scorer receives

The scorer does not read these tables. It receives one feature vector per account:

```json
{"account_ref": "opaque-id", "as_of": "2026-09-30",
 "features": {"utilization": 0.82, "buffer_days": 6.4, "payroll_gap_days": 16, "od_count_3m": 2, "...": null}}
```

- Names must match the bundle's feature list exactly. An unknown name is an error.
- A missing name or `null` means "not available". That is accepted only for features that were sometimes missing in training (deposit features, for card-only customers) and is scored with the feature's missing-value contribution. Features marked `required` in the bundle must always be present; the scorer rejects the request otherwise.
- Every non-dollar feature is rounded to 4 decimal places, dollar features to cents. Compute them the same way.

Feature definitions are in `ml/forewarn/registry.py` and `ml/forewarn/features.py`. A bank will usually reimplement `features.py` in its own warehouse; `ml/tests/test_features.py` holds hand-worked examples to test that reimplementation against.

## Eligibility for a score

An account is scored for a month when it is open and current at the statement date and has at least nine months of history.
