# Results of the last run

Bundle `a2fddd3ce02b00920905b0cdf7823bf3b7a1e44f190e84b8fc6a6812a013919c`. Data source: **synthetic**. 1,284,169 training, 347,246 validation and 497,767 out-of-time account-months (56,555 customers in the out-of-time window).

> Simulated customers. These numbers show the pipeline works and what it measures. They are not evidence about a real portfolio.

## 1. Early warning from the bank's own transactions

| Out-of-time, 60+ days past due within 6 months | AUC | 95% interval | Events in top 5% | 12-month loss $ in top 5% (snapshots with 12 months of outcome) |
|---|---|---|---|---|
| Forewarn (additive, monotone) | 0.903 | 0.897 to 0.910 | 53.6% | 39.3% |
| Additive model, card features only | 0.846 | 0.835 to 0.855 | 47.5% | 34.6% |
| Champion stand-in (card-only scorecard) | 0.828 | 0.816 to 0.838 | 42.4% | 29.0% |
| Unconstrained boosting (benchmark) | 0.910 | 0.904 to 0.917 | 58.4% | 40.3% |

Lead time, measured on 725 accounts whose missed payment led to 60+ days past due and who had missed nothing in the 12 months before:

| Alerted at least | 15 days | 45 days | 75 days | 105 days | 135 days | 165 days | 195 days |
|---|---|---|---|---|---|---|---|
| Forewarn | 69% | 55% | 44% | 36% | 30% | 23% | 17% |
| Champion stand-in, same alert rate | 21% | 14% | 13% | 13% | 12% | 12% | 11% |

A further 314 accounts had a missed payment in the prior 12 months; 89% of those were alerted 75+ days early, but a score that has already seen a miss is not giving early warning, so they are excluded above.

Alert burden: 58 accounts per 1,000 per month; 24.8% of them reach 60+ days past due or file within 6 months; alerts hold 48.7% of 12-month loss dollars (snapshots with 12 months of outcome).

## 2. Recommend the action

35,123 pilot accounts randomized across five arms; value over 18 months; policies cross-fitted and valued by inverse propensity weighting.

| Policy | Gain per account vs no action | 95% interval | Share treated |
|---|---|---|---|
| Date change for all eligible | $40 | -$20 to $103 | 60.5% |
| Hardship plan for all eligible | -$269 | -$343 to -$187 | 100.0% |
| Limit decrease for all eligible | -$129 | -$206 to -$56 | 88.1% |
| Outreach call for everyone | $1 | -$89 to $97 | 100.0% |
| Hand-written playbook | $35 | -$28 to $99 | 36.7% |
| Forewarn policy | $56 | -$22 to $127 | 48.8% |

The rule the scorer ships acts only on alerted accounts inside pilot support. Valued on the 6,484 pilot decisions made after the risk models' training window: -$14 per account (-$127 to $99), acting on 11.2% of them.

## 3. Bankruptcy as a separate target

372 filers; 53.5% were never 60 days past due in the 12 months before filing. Counted per filer: flagged in any of the six snapshots before filing.

| Score | Never-delinquent filers flagged, 2% budget | Never-delinquent filers flagged, 5% budget | All filers flagged, 5% budget |
|---|---|---|---|
| Bankruptcy score | 64.0% | 74.4% | 72.5% |
| Forewarn delinquency score | 31.1% | 52.2% | 62.6% |
| Champion stand-in | 12.0% | 41.3% | 47.1% |

Bankruptcy model overall: AUC 0.886 (0.864 to 0.903); champion stand-in 0.838; unconstrained benchmark 0.905.

## 4. Governance evidence

| Item | Result |
|---|---|
| Cost of the readable model form | 0.007 AUC (delinquency), 0.019 (bankruptcy) against unconstrained boosting |
| Calibration, observed over predicted | 1.08 (delinquency), 0.97 (bankruptcy) |
| Monotone tables | 36 constrained terms checked, 0 violations |
| Reasons per alert | 4.0 on average; 0.0% of alerts with none |
| Group A alert-rate ratio | 1.06 (delinquency), 1.00 (bankruptcy) |
| Largest feature PSI, training to out-of-time | 0.050 (pay_ratio_drift) |
| Restricted feature, AUC it would add | +0.0000 (delinquency), +0.0007 (bankruptcy) |

## 5. Runs inside the bank's account

| Item | Result |
|---|---|
| Bundle size | 66 KiB, 29 input features |
| Go scorer against Python reference | 400 of 400 cases identical; largest logit difference 0.0e+00 |
| Scorer dependencies | Go standard library only |
