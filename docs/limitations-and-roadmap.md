# Limitations and roadmap

## What exists

A working pipeline and scorer, tested end to end on simulated data: features that provably use no future information, two readable risk models, an action policy evaluated on a randomized pilot, a validation report, and a scorer that reproduces the reference implementation exactly.

## What does not exist

| Gap | Why it matters | What closes it |
|---|---|---|
| Real data | Every performance number is a property of the simulator | A design partner's history, analysed inside their environment |
| A real randomized pilot | The action layer cannot be fitted or trusted without one | A pilot run by the bank: a few thousand at-risk accounts per arm, at least twelve months of outcome |
| Bureau attributes | The champion stand-in is weaker than a bank's real behaviour score, so measured lift is flattering | Add the bank's existing score as a comparator and, if the bank wants, as a feature |
| Transaction categorisation | The data contract assumes categories exist | Per-bank mapping from raw transaction codes; expect this to be most of the integration effort |
| Independent validation | The report is developer evidence | The bank's second line, or a validation firm, repeats it |
| Legal review | The regulatory map is an engineer's reading, part of it from memory | Counsel |
| Security review | No penetration test, no SBOM, no signed releases | Standard release engineering |
| Products other than cards | Features and labels are card-specific | New label definitions and a new pilot per product |
| Validated infrastructure code | Kubernetes manifests were syntax-checked only; Terraform was not validated | Run in a real account |

## Modelling limits

- **Additive models.** No interactions. The measured cost against unconstrained boosting is small on simulated data and unknown elsewhere. Pairwise terms can be added while keeping exact reasons; that is the first extension to try if the gap is larger on real data.
- **One calibration window.** Calibration moves under the simulated downturn. Production use needs a recalibration rule.
- **Monthly snapshots.** A mid-month payroll gap is seen at month end. Weekly snapshots would shorten that at the cost of more scoring runs.
- **Six-month horizon.** Chosen to match the early-warning purpose. Longer horizons need longer histories.
- **Action value excludes penalty fees.** This is a deliberate default: a policy that counts late fees as a reason not to help customers pay on time is a conduct problem. It is a parameter, and it changes which actions pay.
- **Recommendations only inside pilot support.** In the shipped run a meaningful share of alerted accounts fall outside it and receive an alert with no recommendation.
- **The recommendation layer has no demonstrated benefit yet.** On the reference run neither the unrestricted policy nor the narrower rule the scorer ships beats doing nothing with any confidence; the shipped rule's point estimate is slightly negative. Treat the action output as experimental until a larger pilot says otherwise.
- **The audit log is a hash chain without a key.** It exposes edits, removals and reordering. It does not stop someone who can rewrite the whole file, and it needs a head hash stored elsewhere to expose truncation. Use write-once storage and keep the batch summary separately.

## Order of work for a first bank

1. Settle intellectual-property ownership and any conflict with current employment before anyone sees the code.
2. Sign a data-processing agreement under which no data leaves the bank.
3. Map the bank's tables to the data contract. Rebuild features in their warehouse and test against `ml/tests/test_features.py`.
4. Backtest on 36 or more months of history. Compare with the bank's own behaviour score at the same alert rate, in dollars of loss reached.
5. Run in shadow for one quarter: scores and reasons produced, nothing acted on.
6. Design the randomized pilot with the bank's collections and compliance teams. Run it.
7. Fit the action layer on pilot outcomes. Validate. Only then recommend actions.

## Commercial questions still open

- **Who the buyer is.** Validation evidence matters most to banks above $30 billion, which are also the ones most able to build this themselves. Smaller lenders feel the model-risk pressure less and have less data. The first customer conversation should test which side of that trade is real.
- **What is being sold.** Software licence, a validation package, or outcomes. The in-account deployment makes usage-based pricing hard to meter.
- **Why a bank would not build it.** The honest answer today is speed and the packaged evidence. That is a head start and not a moat. A cross-bank benchmark or pooled pilot evidence would be one; both conflict with "no data leaves" unless done with care.
