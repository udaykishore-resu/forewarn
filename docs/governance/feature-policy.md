# Feature policy

Every model input is declared in `ml/forewarn/registry.py` with a compliance status. The status is enforced in code at three points: the training pipeline (`assert_permitted`), the export step, and the scorer's bundle loader.

| Status | Meaning | Enforcement |
|---|---|---|
| `allowed` | Used by default | |
| `restricted` | Computed, kept out of production models unless the bank's compliance function signs off | Training excludes it by default. The scorer refuses a bundle that uses one unless started with `-allow-restricted` |
| `prohibited` | Never computed, never accepted | Feature code does not produce it. Training, export and the scorer all reject it. No flag overrides this |

## Prohibited

**`gov_benefit_flag_3m`: whether any deposit came from a government benefit programme.**
Regulation B lists as a prohibited basis "the fact that all or part of the applicant's income derives from any public assistance program" (12 CFR 1002.2(z)). The *amount* of that income counts inside `inflow_ratio_3m`, alongside payroll and other income. The *fact* of receiving it is never a feature.

## Restricted

**`legal_services_3m`: a debit to a legal-services merchant (MCC 8111) in the last three months.**
In the simulator this precedes some bankruptcy filings. Scoring customers on where they spend has drawn regulatory action in the past, and "you paid a lawyer" reads badly in an adverse-action notice. It is excluded by default. The validation report shows what it would add so that a bank can make an informed choice; in the shipped run it adds effectively nothing.

## Rules for adding a feature

1. It is computed from the bank's own records about the account holder's behaviour on their accounts.
2. It is not a prohibited basis and not an obvious stand-in for one. No geography, no names, no age, no merchant categories that reveal health, religion or family status.
3. It has a plain-language reason statement that a customer could act on or at least recognise.
4. It has a declared direction unless there is a stated reason to leave it unconstrained.
5. It is computed as of the snapshot date. `ml/tests/test_point_in_time.py` must still pass.
6. After adding it, the proxy screen in the validation report is reviewed for the new feature.

## Unconstrained features

Four inputs carry no declared direction because either direction can signal trouble: `transfer_in_share_3m`, `ext_debt_pay_change_3m`, `disc_spend_ratio_3m`, `card_spend_ratio_3m`. Their reason statements describe a change and do not say which way. A bank that wants every reason to be directional can drop them; the cost is measurable by retraining.
