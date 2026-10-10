# Operations

## First run, step by step

```bash
make setup          # Python dependencies (macOS: brew install libomp first)
make demo           # 20,000 simulated customers end to end, about a minute
open artifacts/report/validation_report.html
```

`make demo` runs, in order:

| Step | Command | Output |
|---|---|---|
| 1 | `forewarn.cli simulate` | `data/raw/*.parquet`: the synthetic bank |
| 2 | `forewarn.cli features` | `data/work/features.parquet`, `labels.parquet` |
| 3 | `forewarn.cli train` | `artifacts/models/*.json`, scores for every snapshot |
| 4 | `forewarn.cli recommend` | action value models, `artifacts/metrics/policy_evaluation.json` |
| 5 | `forewarn.cli export` | `artifacts/bundle/bundle.json`, `bundle.sha256`, `parity_cases.jsonl` |
| 6 | `forewarn.cli validate` | `artifacts/metrics/validation.json` |
| 7 | `make build` | `bin/forewarn` |
| 8 | `make verify` | `artifacts/metrics/parity.json` |
| 9 | `forewarn.cli report` | `artifacts/report/validation_report.html`, `model_card.md` |

`make all` does the same on 100,000 customers and needs about 5 GB of memory.

## Score one account

```bash
make serve          # listens on 127.0.0.1:8080
```

```bash
curl -s localhost:8080/v1/score -H 'content-type: application/json' \
     -d @artifacts/bundle/example_request.json
```

`example_request.json` is a complete feature vector for one alerted account, written by the export step. The response carries both scores, up to four reasons per score, the recommended action with the estimated gain of every action, and the bundle digest.

Thirteen features are required: the ones that were never missing in training, all of them card or exposure fields (`"required": true` in the bundle, listed by `GET /v1/model`). Deposit features and the two spending ratios may be `null` or left out. A request that omits a required feature, or names a feature the bundle does not know, is rejected with status 400.

| Endpoint | Purpose |
|---|---|
| `POST /v1/score` | One account |
| `POST /v1/score/batch` | Up to 1,000 accounts; rejected as a whole if any item is invalid |
| `GET /v1/model` | Version, digest, thresholds, feature list |
| `GET /v1/drift` | PSI of everything scored since start, against the training reference |
| `GET /metrics` | Prometheus text: decisions, alerts, actions, notices, latency, PSI |
| `GET /healthz`, `GET /readyz` | Probes |

## Score the whole book

```bash
bin/forewarn batch -bundle artifacts/bundle/bundle.json \
  -in features.jsonl -out decisions.jsonl \
  -audit-log audit.jsonl -summary summary.json -fail-on-drift
```

One JSON object per line in, one decision per line out. The run stops at the first malformed line. `summary.json` holds counts and the drift report. Exit status 3 means a score's PSI reached 0.25. Add `-fail-on-feature-drift` to fail on inputs too.

## Checks to run before and after a release

| Check | Command | Fails when |
|---|---|---|
| Scorer reproduces the reference | `make verify` | Any score, alert, reason or action differs |
| Bundle is the approved one | start with `-bundle-sha256 <digest>` | The file on disk has a different digest |
| Bundle keeps its promises | automatic at load | Prohibited or restricted feature, non-monotone table, malformed tables |
| No outbound client in the scorer | `make no-egress-check` | An HTTP client, dialer or subprocess appears in scorer source |
| Audit log is intact | `bin/forewarn audit-verify -file audit.jsonl -expect-head <audit_head from the batch summary>` | A record was edited, removed or reordered, or the end of the log is not the one recorded. Without `-expect-head` or `-expect-records`, records dropped from the end go unnoticed |
| The model reads as documented | `bin/forewarn inspect -bundle ...` | Reviewed by a person |

## Monitoring

| Signal | Where | Suggested response |
|---|---|---|
| Score PSI ≥ 0.10 | batch summary, `/metrics` | Find the inputs that moved and why |
| Score PSI ≥ 0.25 | batch exits 3 | Hold the scores until explained |
| Input PSI ≥ 0.25 with score PSI low | batch summary | Usually seasonal or calendar (see below); check categorisation and pipeline changes |
| Alert rate away from the validation share | `forewarn_alerts_total / forewarn_decisions_total` | Usually the same cause as a score PSI move |
| Observed over predicted outside the bank's band | monthly, from matured outcomes | Recalibrate (two numbers in the bundle), then revalidate |
| Out-of-time AUC below the bank's floor | quarterly | Retrain |
| Share of alerts outside pilot support rising | decision notes | The pilot population no longer matches the book; plan a new pilot |

**Single months and seasonality.** The reference bins pool nineteen months. One December batch, compared with that pool, shows four inputs above 0.25 (spending ratios and utilization changes, from holiday spending) while both score PSIs stay under 0.07. The validation report has the month-by-month table. This is why only score drift fails a run by default. A bank that wants tighter input monitoring should keep a reference per calendar month.

Outcome-based monitoring needs matured labels: six months for the scores, eighteen for action value. The pipeline's `validate` step is the tool for it; run it on the bank's own outcomes each quarter.

## Deploying inside a bank's account

1. Mirror the two base images and the Python packages into the bank's registries.
2. Build the image: `make image`. Push it to the private registry.
3. Apply the perimeter in `deploy/terraform/aws` (reference; review and validate first).
4. Copy the approved `bundle.json` to `deploy/k8s/`, put its digest in `kustomization.yaml`, set the image, and `kubectl apply -k deploy/k8s`.
5. Label the calling namespace `forewarn.io/client=true`.
6. Run `forewarn verify` inside the cluster against the mounted bundle as a release gate.

## Changing a model

A new bundle is a model change. The digest changes, the pinned value in the deployment must change with it, and the validation report must be regenerated for that digest. The scorer logs the digest at start and writes it into every decision, so any decision can be traced to the exact tables that produced it.
