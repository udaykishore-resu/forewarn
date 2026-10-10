# Forewarn model card

Bundle `a2fddd3ce02b00920905b0cdf7823bf3b7a1e44f190e84b8fc6a6812a013919c` · data source: **synthetic**

> Trained and evaluated on simulated customers. Not for decisions about real customers.

## What it does

| Output | Definition |
|---|---|
| Delinquency score | Probability an account that is current today is 60+ days past due within 6 months |
| Bankruptcy score | Probability an account that is current today files for bankruptcy within 6 months |
| Reasons | Up to four feature groups adding the most risk relative to the average account, exact from the scoring tables |
| Recommended action | Payment-date change, hardship plan, limit decrease, outreach call, or none |

## Inputs

27 features computed from the bank's own deposit transactions, card transactions and card statements. No bureau file, no data from another institution.

| Feature | Source | Direction | Reason statement |
|---|---|---|---|
| `payroll_gap_days` | deposit | risk rises with value | Length of time since the last regular payroll deposit |
| `payroll_amt_ratio_3m` | deposit | risk falls with value | Decrease in regular payroll deposits |
| `payroll_missed_cycles_3m` | deposit | risk rises with value | Expected payroll deposits not received |
| `inflow_ratio_3m` | deposit | risk falls with value | Decrease in total income deposited to the account |
| `transfer_in_share_3m` | deposit | unconstrained | Share of deposits arriving by transfer from other accounts |
| `buffer_days` | deposit | risk falls with value | Deposit balance low relative to monthly spending |
| `buffer_days_change_6m` | deposit | risk falls with value | Decline in deposit balance relative to spending |
| `min_balance_ratio_3m` | deposit | risk falls with value | Low minimum deposit balance |
| `od_count_3m` | deposit | risk rises with value | Number of recent overdrafts |
| `od_months_6m` | deposit | risk rises with value | Frequency of overdrafts |
| `neg_days_3m` | deposit | risk rises with value | Number of days with a negative deposit balance |
| `stl_months_6m` | deposit | risk rises with value | Recent short-term loan activity |
| `ext_debt_pay_ratio_3m` | deposit | risk rises with value | Payments to other lenders high relative to income deposited |
| `ext_debt_pay_change_3m` | deposit | unconstrained | Change in payments to other lenders |
| `disc_spend_ratio_3m` | card | unconstrained | Change in discretionary spending |
| `utilization` | card | risk rises with value | Balance high relative to credit limit |
| `util_change_3m` | card | risk rises with value | Recent increase in balance relative to credit limit |
| `util_change_6m` | card | risk rises with value | Increase in balance relative to credit limit over six months |
| `pay_ratio_3m` | card | risk falls with value | Payments low relative to balance |
| `pay_to_min_3m` | card | risk falls with value | Payments close to the minimum amount due |
| `min_pay_months_6m` | card | risk rises with value | Number of months only the minimum payment was made |
| `pay_ratio_drift` | card | risk falls with value | Decrease in payment amounts |
| `cash_adv_to_limit_3m` | card | risk rises with value | Amount of recent cash advances |
| `cash_adv_months_6m` | card | risk rises with value | Frequency of recent cash advances |
| `late_fee_count_6m` | card | risk rises with value | Recent late payments |
| `card_spend_ratio_3m` | card | unconstrained | Change in card spending |
| `max_cpd_12m` | card | risk rises with value | Past delinquency on this account |

Excluded by policy: `legal_services_3m` (restricted, merchant-category behaviour) and `gov_benefit_flag_3m` (prohibited basis under Regulation B).

## Model form

Gradient-boosted depth-one trees with monotone constraints, folded into one lookup table per feature. The score is an intercept plus one table value per feature, passed through a logistic calibration.

## Out-of-time performance

| | Delinquency | Bankruptcy |
|---|---|---|
| AUC | 0.903 | 0.886 |
| 95% interval | 0.897 to 0.910 | 0.864 to 0.903 |
| Champion stand-in AUC | 0.828 | 0.838 |
| Unconstrained benchmark AUC | 0.910 | 0.905 |
| Observed / predicted | 1.08 | 0.97 |
| Alert threshold (probability) | 0.0996 | 0.0386 |

- Alerted at least 75 days before the first missed payment: 44.1% of accounts that went on to 60+ days past due after a year with no missed payment.
- Filers never 60 days past due in the 12 months before filing: 53.5%. Flagged before filing at a 5% budget by the bankruptcy score: 74.4%; by the delinquency score: 52.2%.
- Action policy gain over no action, cross-fitted on the randomized pilot: $56 per account (95% interval -$22 to $127). The rule the scorer ships: -$14 (-$127 to $99).

## Limits of use

- Current, open revolving card accounts only. Not for origination, pricing or collections.
- A limit decrease on a current account is adverse action under Regulation B; the scorer marks those decisions and supplies the reasons.
- Recommendations are advisory. A person or a governed decision process acts on them.
- Recalibrate when observed over predicted leaves the band the bank sets. Investigate when a score's PSI reaches 0.25; single inputs cross that level seasonally without the scores moving.
- The action recommendation has no demonstrated benefit on the reference run and should be treated as experimental.

Full evidence: `validation_report.html`.
