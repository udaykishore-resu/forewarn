# United States regulatory map

Scope: federal rules that bear on scoring existing credit card accounts and acting on the scores. State law is out of scope and matters, particularly for fair lending.

This is an engineering document. It records which rule each control answers and how far each statement was checked. It is not legal advice, and a bank's counsel and compliance function decide what applies.

**How to read the "Checked" column**

| Mark | Meaning |
|---|---|
| Read | The regulation or agency text was read while building this (October 2026) |
| Summary | Taken from law-firm or consultancy summaries of the primary text |
| Recalled | Written from the author's knowledge and not re-checked. Verify before relying on it |

## 1. Model risk management

| Point | Checked |
|---|---|
| The Federal Reserve, OCC and FDIC issued revised model risk guidance on April 17, 2026 (Fed SR 26-2, OCC Bulletin 2026-13, FDIC FIL-15-2026). It supersedes SR 11-7 and SR 21-8. | Read (Fed cover letter) |
| The Fed's letter says the guidance "is expected to be most relevant to banking organizations with over $30 billion in total assets." | Read |
| It keeps three areas: model development and use, validation and monitoring (including conceptual soundness and outcomes analysis), and governance and controls. | Summary |
| A bank stays responsible for model risk when the model comes from a vendor. | Summary |
| Generative and agentic AI are outside its scope. | Summary |
| The guidance attachment itself was **not** read; only the cover letter and summaries were. | — |

Consequences for the product:

- The "validation evidence in the box" argument is strongest at banks above $30 billion. Credit unions are supervised by the NCUA, and smaller banks sit below the stated threshold. They still validate vendor models, but the pressure is lower.
- Forewarn uses no generative AI anywhere in the scoring path, so the whole decision falls under the conventional guidance.

| What a validator asks for | What ships |
|---|---|
| Purpose, intended use, limits of use | Model card; "Scope" section of the validation report |
| Development data, target definitions, exclusions | `docs/data-contract.md`; windows and base rates in the report |
| Conceptual soundness | Additive monotone form; term tables and shapes; direction check; feature policy |
| Outcomes analysis | Out-of-time discrimination with intervals, capture, calibration by decile, month-by-month stability |
| Benchmarking | Champion stand-in, card-only model, unconstrained boosting on the same rows |
| Sensitivity | Restricted-feature ablation; drop-feature refits in the fair-lending section |
| Ongoing monitoring | PSI per feature and per score from reference bins in the bundle; `forewarn batch -fail-on-drift`; `/metrics` |
| Implementation testing | `forewarn verify` replays reference cases; result recorded in the report |
| Change control | Bundle identified by SHA-256; scorer refuses a bundle whose digest differs from the pinned one |
| Audit trail | Hash-chained decision log; `forewarn audit-verify` |
| Limitations | Listed in the report and in `docs/limitations-and-roadmap.md` |

What a bank must still do itself: independent validation by its own second line, a model inventory entry, approval, and ongoing performance review on its own outcomes. Evidence from a vendor shortens that work. It does not replace it.

## 2. Equal Credit Opportunity Act and Regulation B (12 CFR 1002)

| Point | Checked |
|---|---|
| Adverse action includes "a termination of an account or an unfavorable change in the terms of an account" that does not affect all or substantially all of a class of accounts (1002.2(c)(1)(ii)). | Read |
| It does not include a change the customer expressly agreed to, or action taken in connection with inactivity, default or delinquency on that account (1002.2(c)(2)). | Read |
| Notice is due within 30 days after adverse action on an existing account (1002.9(a)(1)(iii)), in writing, with the specific principal reasons or the right to request them (1002.9(a)(2), (b)(2)). | Read |
| "Failed to achieve a qualifying score" and "internal standards" are not sufficient reasons (1002.9(b)(2)). | Read |
| Official commentary: more than four reasons is unlikely to help the applicant (9(b)(2)-1); reasons must relate to factors actually scored and no principal reason may be left out (9(b)(2)-4); acceptable methods include ranking factors by how far the applicant fell below the average for all applicants (9(b)(2)-5). | Read |
| Prohibited bases include race, colour, religion, national origin, sex, marital status, age, receipt of public assistance income, and good-faith exercise of Consumer Credit Protection Act rights (1002.2(z)). | Read |
| A scoring system should be periodically revalidated (1002.2(p)(1)(iv)). | Read |
| Amendments published April 22, 2026 and effective July 21, 2026 state that ECOA does not authorise disparate-impact liability. Disparate treatment, including neutral criteria designed or applied as proxies for a prohibited basis, remains prohibited. State fair-lending law and the Fair Housing Act may still reach disparate impact. | Summary |

How Forewarn answers this:

- **A limit decrease on a current account is adverse action.** The accounts Forewarn scores are current by definition, so the delinquency exclusion does not apply. The scorer sets `adverse_action_notice: true` on every such recommendation and attaches the reasons.
- **Reasons come from the score itself.** Each reason is a group of features whose table values added the most risk relative to the average account, which is the second method in comment 9(b)(2)-5. At most four are returned.
- **A payment-date change, a hardship plan the customer accepts, and an outreach call** are not unfavourable changes imposed on the customer. Whether a particular hardship plan's terms amount to one is a question for counsel.
- **Public assistance income** is handled in `feature-policy.md`: its amount counts as income, its presence is never a feature, and the scorer rejects any bundle that says otherwise.
- **Fair-lending tests still ship** after the 2026 amendments: alert-rate ratios, accuracy and calibration by group, a proxy screen, and a search for less discriminatory alternatives. Proxy use is still prohibited, state law may differ, and bank policy usually requires the tests regardless.

## 3. Fair Credit Reporting Act

| Point | Checked |
|---|---|
| FCRA adverse-action duties attach when action is based on information in a consumer report. | Recalled |
| A report containing only the reporting company's own transactions or experiences with the consumer is excluded from the definition of a consumer report. | Recalled |

Forewarn uses only the bank's own records, so by design its decisions are not "based on a consumer report." If a bank adds bureau attributes to the feature set, FCRA notice and score-disclosure duties come into play and the reason logic has to carry the bureau factors too.

## 4. Truth in Lending Act and Regulation Z (12 CFR 1026)

All recalled and not re-checked. Section numbers are the author's best recollection.

| Action | Rule to confirm |
|---|---|
| Limit decrease | Notice of the decrease is required, and an over-the-limit fee or penalty rate may not be imposed for exceeding the new limit until 45 days after notice (1026.9(c)(2)(vi)) |
| Hardship plan | Temporary hardship and workout arrangements have their own notice treatment (1026.9(c)(2)(v)(D)) and rate rules on completion or failure (1026.55(b)(5)) |
| Payment-date change | Due dates must fall on the same day each month (1026.7(b)(11)); a change at the customer's request is ordinary servicing |

The scorer recommends. The bank's servicing systems carry out the action and send the notices.

## 5. Unfair, deceptive or abusive acts or practices

Recalled. Acting on behavioural data in ways customers would not expect has been challenged as deceptive when it was not disclosed; a 2008 FTC action against a subprime card issuer over undisclosed behavioural scoring is the usual reference. This is why merchant-category features are restricted by default and why reasons are written in terms a customer can recognise.

## 6. Bankruptcy

Recalled. Once a customer files, the automatic stay (11 U.S.C. § 362) bars collection activity. The bankruptcy score applies only to accounts that are open and current, before any filing, and the bank's existing bankruptcy handling takes over at filing.

Reducing a credit line because a model predicts a filing is lawful account management in general but is exactly the kind of action that draws complaints. Three safeguards are built in: the limit-decrease recommendation is learned from randomized outcomes and not assumed; it requires a minimum estimated gain; and it always carries reasons. A bank may reasonably decide to drop that arm. That means re-exporting the bundle without it, which yields a new digest and goes through change control like any other model change.

## 7. Data handling and third-party risk

Recalled. Interagency guidance on third-party relationships (2023) expects banks to manage vendor risk across the relationship. Forewarn's design removes the largest item from that review: the vendor never receives customer data, because training, validation and scoring all run inside the bank's account with no route out. What remains for the bank to assess is the software supply chain. The scorer has no third-party code; the training pipeline depends on a small set of widely used Python packages that the bank mirrors internally.

## Sources

- Federal Reserve, SR 26-2, Revised Guidance on Model Risk Management: https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm
- CFPB, Regulation B § 1002.2 Definitions: https://www.consumerfinance.gov/rules-policy/regulations/1002/2/
- CFPB, Regulation B § 1002.9 Notifications, with official interpretations: https://www.consumerfinance.gov/rules-policy/regulations/1002/9/
- Cooley, "CFPB Finalizes Significant Changes to Regulation B": https://finsights.cooley.com/cfpb-finalizes-significant-changes-to-regulation-b/
- Baker Tilly, "Updated interagency guidance on model risk management": https://www.bakertilly.com/insights/updated-interagency-guidance-on-model-risk-management
- Elevate Consult, "Model Risk Management Framework: What to Rebuild After 2026": https://elevateconsult.com/insights/model-risk-management-framework-rebuild-revised-guidance/
- Domino Data Lab, "SR 26-2": https://domino.ai/data-science-dictionary/sr-26-2
