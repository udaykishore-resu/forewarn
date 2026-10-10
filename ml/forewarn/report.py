"""Render the validation report (one self-contained HTML file) and the model card."""
from __future__ import annotations

import datetime as dt
from html import escape
from pathlib import Path

from . import charts as C
from .registry import FEATURES
from .store import Store

CSS = """
:root{--surface:#fcfcfb;--panel:#f4f3f0;--line:#e3e2dd;--text:#0b0b0b;--text2:#52514e;--muted:#807f7a;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--mute:#c9c8c2;--warn-bg:#fff4dc;--warn-line:#e0a100;--ok:#0a7d0a;--bad:#c23030}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){--surface:#1a1a19;--panel:#242422;--line:#383835;
--text:#f0efec;--text2:#c3c2b7;--muted:#96958d;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--mute:#55544f;--warn-bg:#3a2f12;--warn-line:#c98500;--ok:#4cc24c;--bad:#e66767}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--line:#383835;--text:#f0efec;--text2:#c3c2b7;--muted:#96958d;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--mute:#55544f;--warn-bg:#3a2f12;--warn-line:#c98500;--ok:#4cc24c;--bad:#e66767}
*{box-sizing:border-box}body{margin:0;background:var(--surface);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:28px;line-height:1.2;margin:0 0 6px}h2{font-size:20px;margin:44px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:22px 0 6px}p{margin:8px 0;max-width:78ch}.sub{color:var(--text2)}.small{font-size:13px;color:var(--text2)}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--panel);padding:1px 4px;border-radius:4px}
.banner{background:var(--warn-bg);border-left:4px solid var(--warn-line);padding:12px 14px;border-radius:6px;margin:16px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--panel);border-radius:10px;padding:14px}.tile .k{font-size:13px;color:var(--text2)}
.tile .v{font-size:26px;font-weight:600;margin:2px 0}.tile .d{font-size:12.5px;color:var(--text2)}
.wrap{overflow-x:auto}table{border-collapse:collapse;margin:10px 0;font-size:13.5px;min-width:480px}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:150px}th{color:var(--text2);font-weight:600}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(290px,100%),1fr));gap:14px}.grid2>*{min-width:0}
code{overflow-wrap:anywhere}
.chart{width:100%;height:auto;max-width:680px;display:block}.grid2 .chart{max-width:340px}
.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{stroke:var(--muted);stroke-width:1}
.chart .ref{stroke:var(--muted);stroke-width:1}.chart .whisker{stroke:var(--text);stroke-width:1.5}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .dot{stroke:var(--surface);stroke-width:2}.chart .tick{font-size:11px;fill:var(--text2)}
.chart .lbl{font-size:12px;fill:var(--text)}.chart .val{font-size:11.5px;fill:var(--text)}.chart .strong{font-weight:600}
.s1.line{stroke:var(--s1)}.s2.line{stroke:var(--s2)}.s3.line{stroke:var(--s3)}
.s1.dot,.s1.fill{fill:var(--s1)}.s2.dot,.s2.fill{fill:var(--s2)}.s3.dot,.s3.fill{fill:var(--s3)}.mute.fill{fill:var(--mute)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
nav{font-size:13px;color:var(--text2);margin:10px 0 0;display:flex;flex-wrap:wrap;gap:4px 14px}nav a{color:var(--s1);white-space:nowrap}
"""


def pct(v, d=1):
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def num(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def usd(v):
    return "n/a" if v is None else (f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")


def table(head: list[str], rows: list[list], caption: str = "") -> str:
    h = "".join(f"<th>{escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c if str(c).startswith('<') else escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    cap = f'<p class="small">{escape(caption)}</p>' if caption else ""
    return f'<div class="wrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>{cap}'


def tile(k, v, d=""):
    return f'<div class="tile"><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div><div class="d">{escape(d)}</div></div>'


LABELS = {"forewarn": "Forewarn (additive, monotone)", "champion": "Champion stand-in (card-only scorecard)",
          "card_only": "Additive model, card features only", "benchmark_unconstrained": "Unconstrained boosting (benchmark)",
          "with_restricted_feature": "Forewarn + restricted feature"}
ARM_LABEL = {"none": "No action", "due_date_change": "Payment-date change", "hardship_plan": "Hardship plan",
             "limit_decrease": "Limit decrease", "outreach_call": "Outreach call"}
POLICY_LABEL = {"no_action": "No action for anyone", "always_due_date_change": "Date change for all eligible",
                "always_hardship_plan": "Hardship plan for all eligible", "always_limit_decrease": "Limit decrease for all eligible",
                "always_outreach_call": "Outreach call for everyone", "expert_rules": "Hand-written playbook", "forewarn_policy": "Forewarn policy"}


def _model_section(name: str, m: dict, top_key: str) -> str:
    rows = []
    for key, lab in LABELS.items():
        c = m["comparators"][key]
        t = c[top_key]
        rows.append([lab, num(c["auc"], 4), f"{c['auc_ci95'][0]:.4f} to {c['auc_ci95'][1]:.4f}", num(c["ks"]),
                     pct(t["event_capture"]), pct(t["precision"]), f"{t['lift']:.1f}x", pct(t.get("amount_capture"))])
    share = top_key.split("_")[1].replace("pct", "%")
    out = [table(["Model", "AUC", "95% interval", "KS", f"Events in top {share}", f"Precision in top {share}", "Lift", "12-month loss $ captured"], rows,
                 "Out-of-time window. Intervals resample whole customers (200 draws for the first two rows, 50 for the rest). "
                 "Loss dollars use only the snapshots that have a full 12 months of outcome after them.")]
    cal = m["calibration_test"]
    months = m["by_month"]
    out.append('<div class="grid2"><div>')
    out.append(f"<h3>Calibration by score decile</h3>{C.calibration_chart([r['predicted'] for r in cal], [r['observed'] for r in cal], f'{name} calibration')}")
    out.append(f'<p class="small">Observed over predicted across the window: <b>{m["observed_over_predicted_test"]:.2f}</b>. Points on the line are perfectly calibrated.</p></div><div>')
    out.append("<h3>Predicted and observed by decile</h3>")
    out.append(table(["Decile", "Predicted", "Observed", "Obs / pred"], [[r["bin"], pct(r["predicted"], 2), pct(r["observed"], 2), num(r["observed_over_predicted"], 2)] for r in cal]))
    out.append("</div></div>")
    out.append("<h3>Discrimination month by month</h3>")
    out.append(C.line_chart([str(r["month"]) for r in months],
                            [("Forewarn", [r["auc"] for r in months]), ("Benchmark", [r["auc_benchmark"] for r in months]), ("Champion", [r["auc_champion"] for r in months])],
                            f"{name} AUC by snapshot month", y_fmt=lambda v: f"{v:.2f}", x_title="snapshot month (out-of-time window)"))
    out.append(table(["Month", "Accounts", "AUC", "Champion AUC", "Predicted", "Observed", "Alert rate"],
                     [[r["month"], f"{r['n']:,}", num(r["auc"]), num(r["auc_champion"]), pct(r["predicted"], 2), pct(r["observed"], 2), pct(r["alert_rate"], 2)] for r in months]))
    seg = m["segments"]
    out.append("<h3>By data availability</h3>")
    out.append(table(["Segment", "Share of accounts", "Forewarn AUC", "Champion AUC"],
                     [["Customers with deposit history at the bank", pct(seg["with_deposit_history"]["share"]), num(seg["with_deposit_history"]["auc"]), num(seg["with_deposit_history"]["auc_champion"])],
                      ["Card-only customers (deposit features missing)", pct(seg["card_only_customers"]["share"]), num(seg["card_only_customers"]["auc"]), num(seg["card_only_customers"]["auc_champion"])]]))
    return "".join(out)


def _soundness(name: str, m: dict) -> str:
    arrow = {1: "risk rises", -1: "risk falls", 0: "unconstrained"}
    rows = [[t["feature"], t["group"], arrow[t["declared_direction"]],
             {True: '<span class="ok">holds</span>', False: '<span class="bad">violated</span>', None: "n/a"}[t["monotone_ok"]],
             num(t["importance"]), num(t["range_logit"], 2), t["bins"], f"{t['missing_contribution']:+.2f}"] for t in m["terms"]]
    out = [table(["Feature", "Reason group", "Declared direction", "Table check", "Importance", "Range (log-odds)", "Bins", "Missing"], rows,
                 "Importance is the average absolute contribution in the out-of-time window. Missing is the contribution when the value is absent.")]
    if m["features_unused"]:
        out.append(f'<p class="small">Offered to the model and not used: {escape(", ".join(m["features_unused"]))}.</p>')
    out.append('<div class="grid2">')
    for t in m["terms"][:6]:
        out.append(C.step_chart(t["shape"]["thresholds"], t["shape"]["values"], t["p01"], t["p99"], t["feature"], t["missing_contribution"]))
    out.append('</div><p class="small">Contribution in log-odds across the 1st to 99th percentile of each feature, six most important terms. Zero is the training-population average.</p>')
    return "".join(out)


def render_html(v: dict) -> str:
    md, mb = v["models"]["delinquency"], v["models"]["bankruptcy"]
    cd, cb = md["comparators"], mb["comparators"]
    lt, sep, pol = v["lead_time"], v["separate_target"], v["policy"]
    fw, ch = lt["models"]["forewarn_delinquency"], lt["models"]["champion_delinquency"]
    fp = pol["policies"]["forewarn_policy"]
    par = v.get("implementation_verification")
    synthetic = v["data_source"] == "synthetic"
    o: list[str] = []
    o.append(f"<h1>Forewarn validation report</h1><p class='sub'>Model bundle <code>{v['bundle_sha256'][:16]}…</code> · generated {dt.date.today().isoformat()}</p>")
    o.append('<nav><a href="#outcomes">Outcomes</a><a href="#soundness">Conceptual soundness</a><a href="#lead">Lead time</a><a href="#bk">Bankruptcy target</a>'
             '<a href="#reasons">Reasons</a><a href="#actions">Actions</a><a href="#fair">Fair lending</a><a href="#drift">Drift</a><a href="#impl">Implementation</a><a href="#limits">Limitations</a></nav>')
    if synthetic:
        o.append('<div class="banner"><b>Every number in this report comes from simulated customers.</b> The simulator encodes the distress mechanisms the models then find, '
                 'so these results show that the pipeline measures the right things and that the two implementations agree. They are not evidence of performance on a real portfolio, '
                 'and this bundle must not be used for decisions about real customers.</div>')
    o.append('<div class="tiles">')
    o.append(tile("Delinquency AUC, out of time", num(cd["forewarn"]["auc"]), f"champion stand-in {num(cd['champion']['auc'])}; unconstrained benchmark {num(cd['benchmark_unconstrained']['auc'])}"))
    o.append(tile("Bankruptcy AUC, out of time", num(cb["forewarn"]["auc"]), f"champion stand-in {num(cb['champion']['auc'])}; unconstrained benchmark {num(cb['benchmark_unconstrained']['auc'])}"))
    o.append(tile("Alerted 75+ days before first miss", pct(fw["share_with_lead_at_least"]["75"], 0), f"accounts with a clean prior year; champion stand-in at the same alert rate: {pct(ch['share_with_lead_at_least']['75'], 0)}"))
    fb = sep["flagged_before_filing"]
    o.append(tile("Never-delinquent filers flagged before filing", pct(fb["bankruptcy_score_top5pct"]["clean_filers"], 0), f"delinquency score at the same 5% budget: {pct(fb['delinquency_score_top5pct']['clean_filers'], 0)}"))
    inconclusive = fp["gain_ci95"][0] <= 0
    srt = pol["shipped_rule"]
    o.append(tile("Policy gain per pilot account", usd(fp["gain_vs_no_action"]),
                  f"95% interval {usd(fp['gain_ci95'][0])} to {usd(fp['gain_ci95'][1])} versus no action" + ("; includes zero" if inconclusive else "")
                  + f". Rule the scorer ships: {usd(srt['gain_vs_no_action'])} ({usd(srt['gain_ci95'][0])} to {usd(srt['gain_ci95'][1])})"))
    if par:
        o.append(tile("Go scorer vs reference", f"{par['passed']}/{par['cases']}", f"largest logit difference {par['max_abs_diff_logit']:.1e}"))
    o.append("</div>")

    o.append("<h2>Scope</h2>")
    o.append("<p><b>Intended use.</b> Monthly scoring of open credit card accounts that are current, using the bank's own deposit and card records, to (1) flag accounts likely to reach 60 or more days past due within six months, "
             "(2) flag accounts likely to file for bankruptcy within six months, and (3) recommend one of four account-management actions, or none, for flagged accounts. "
             "Recommendations go to a person or a governed decision process; the scorer changes nothing on an account.</p>"
             "<p><b>Not intended for</b> origination or line-increase decisions, accounts that are already past due, collections prioritisation, pricing, or any product other than revolving card accounts.</p>")
    w = v["windows"]
    o.append(table(["Window", "Snapshot months", "Account-months", "60+ DPD rate", "Bankruptcy rate"],
                   [[s.capitalize(), f"{w[s][0]} to {w[s][1]}", f"{v['rows'][s]:,}", pct(v["base_rates"][s]["delinquency"], 2), pct(v["base_rates"][s]["bankruptcy"], 2)] for s in ("train", "valid", "test")],
                   "Validation rows set early stopping, calibration and alert thresholds. Six months of snapshots between validation and test are discarded: that is how long validation labels take to mature. "
                   "Rows affected by a pilot action are excluded and control-arm rows are weighted up, so the scores describe what happens when the bank does nothing."))

    o.append('<h2 id="outcomes">Outcomes analysis and benchmarking</h2>')
    o.append(f"<p>The additive model gives up {cd['benchmark_unconstrained']['auc'] - cd['forewarn']['auc']:.3f} AUC on delinquency and {cb['benchmark_unconstrained']['auc'] - cb['forewarn']['auc']:.3f} on bankruptcy "
             f"against unconstrained boosting on the same features. That is the measured price of exact reasons and enforced monotonicity. "
             f"Deposit-transaction features add {cd['forewarn']['auc'] - cd['card_only']['auc']:.3f} AUC on delinquency over the same model restricted to card features. "
             f"The champion stand-in has no bureau data, so a bank's real behaviour score would be stronger than the one shown.</p>")
    o.append("<h3>Delinquency: 60+ days past due within 6 months</h3>" + _model_section("delinquency", md, "top_5pct"))
    o.append("<h3>Bankruptcy: filing within 6 months</h3>" + _model_section("bankruptcy", mb, "top_2pct"))
    ab = v["alert_burden"]
    o.append("<h3>Alert burden at the shipped thresholds</h3>")
    o.append(table(["Measure", "Value"], [
        ["Accounts alerted per 1,000 per month (either score)", f"{ab['alerts_per_1000_accounts_per_month']:.0f}"],
        ["Alerted accounts that reach 60+ DPD or file within 6 months", pct(ab["precision_bad_within_6m"])],
        ["Alerted accounts that miss at least one payment or file within 6 months", pct(ab["precision_missed_payment_or_bankruptcy_within_6m"])],
        ["Share of all 60+ DPD and bankruptcy outcomes that were alerted", pct(ab["recall_bad_within_6m"])],
        [f"Share of 12-month credit-loss dollars sitting in alerted accounts (snapshots {ab['loss_dollar_snapshot_months'][0]} to {ab['loss_dollar_snapshot_months'][1]}, which have 12 months of outcome)", pct(ab["share_of_12m_loss_dollars_in_alerts"])],
        ["Base rate of 60+ DPD or bankruptcy within 6 months", pct(ab["base_rate_bad"], 2)],
    ]))

    o.append('<h2 id="soundness">Conceptual soundness</h2>')
    o.append("<p>Both scores are sums of one lookup table per feature. A table is the fold of every depth-one tree that split on that feature, so nothing is approximated when the model is read, scored or explained. "
             "Each feature carries a declared direction in the feature registry; training enforces it, this report re-checks it, and the scorer refuses to load a bundle that breaks it.</p>")
    o.append("<h3>Delinquency model terms</h3>" + _soundness("delinquency", md))
    o.append("<h3>Bankruptcy model terms</h3>" + _soundness("bankruptcy", mb))
    r_d, r_b = cd["with_restricted_feature"]["auc"] - cd["forewarn"]["auc"], cb["with_restricted_feature"]["auc"] - cb["forewarn"]["auc"]
    o.append(f"<h3>Restricted feature</h3><p>Adding the legal-services merchant-category flag changes AUC by {r_d:+.4f} (delinquency) and {r_b:+.4f} (bankruptcy). It is excluded from the shipped models. "
             "A feature flagging receipt of government benefits is prohibited in the registry and is never computed.</p>")

    o.append('<h2 id="lead">Lead time before the first missed payment</h2>')
    rm_fw, rm_ch = lt["recent_miss"]["forewarn_delinquency"], lt["recent_miss"]["champion_delinquency"]
    o.append(f"<p>{lt['events']:,} accounts missed a payment in months {lt['first_miss_months'][0]} to {lt['first_miss_months'][1]}, went on to 60+ days past due, and had not missed any payment in the {lt['clean_months']} months before. "
             f"For each, the {lt['lookback_months']} monthly snapshots before the miss were checked for an alert. "
             "Lead time runs from the snapshot date to the middle of the month of the miss, so the latest possible alert counts as 15 days. Both scores alert on the same share of accounts.</p>")
    b = [str(x) for x in lt["buckets_days"]]
    o.append(C.column_chart([f"{x}+" for x in b], [("Forewarn", [fw["share_with_lead_at_least"][x] for x in b]), ("Champion stand-in", [ch["share_with_lead_at_least"][x] for x in b])],
                            "share of eventual delinquents alerted at least this many days early", y_fmt=lambda x: f"{100 * x:.0f}%", y_max=1.0, x_title="days before first missed payment"))
    o.append(table(["Alerted at least", *[f"{x} days" for x in b], "Median when alerted"],
                   [["Forewarn", *[pct(fw["share_with_lead_at_least"][x]) for x in b], f"{fw['median_lead_days_when_alerted']:.0f} days"],
                    ["Champion stand-in", *[pct(ch["share_with_lead_at_least"][x]) for x in b], f"{ch['median_lead_days_when_alerted']:.0f} days"]]))
    if "lead_75d_plus_job_loss" in fw:
        o.append(table(["Simulated cause of distress", "Events", "Forewarn 75+ days early", "Champion 75+ days early"],
                       [["Job loss in the prior 14 months", f"{fw['n_job_loss']:,}", pct(fw["lead_75d_plus_job_loss"]), pct(ch["lead_75d_plus_job_loss"])],
                        ["Everything else", f"{fw['n_other_causes']:,}", pct(fw["lead_75d_plus_other_causes"]), pct(ch["lead_75d_plus_other_causes"])]],
                       "Cause comes from simulator ground truth, which no model sees."))
    o.append(f"<h3>Accounts that had already missed a payment in the prior {lt['clean_months']} months</h3>"
             f"<p>{lt['events_with_recent_miss']:,} further accounts fit the same definition except that they had a recent missed payment on record. "
             "Both scores see that directly, so an early alert there says little about early warning. They are kept out of the headline.</p>")
    o.append(table(["Alerted at least", *[f"{x} days" for x in b]],
                   [["Forewarn", *[pct(rm_fw["share_with_lead_at_least"][x]) for x in b]],
                    ["Champion stand-in", *[pct(rm_ch["share_with_lead_at_least"][x]) for x in b]]]))

    o.append('<h2 id="bk">Bankruptcy as a separate target</h2>')
    au = sep["auc_account_months"]
    o.append(f"<p>{sep['filers']:,} customers filed in months {sep['filing_months'][0]} to {sep['filing_months'][1]}. {pct(sep['share_never_60dpd_in_prior_12m'])} of them were never 60 days past due in the 12 months before filing. "
             "A delinquency model sees those accounts labelled good right up to the filing, so it learns to rank them low. "
             "The table counts, per filer, whether a score flagged the account in any of the six monthly snapshots before the filing, with each score given the same share of accounts to flag.</p>")
    rows = []
    for label, key in (("Bankruptcy score", "bankruptcy_score"), ("Forewarn delinquency score", "delinquency_score"), ("Champion stand-in (delinquency)", "champion_delinquency")):
        rows.append([label, pct(fb[f"{key}_top2pct"]["clean_filers"]), pct(fb[f"{key}_top5pct"]["clean_filers"]), pct(fb[f"{key}_top5pct"]["all_filers"]), num(au[key])])
    o.append(table(["Score", "Never-delinquent filers flagged, 2% budget", "Never-delinquent filers flagged, 5% budget", "All filers flagged, 5% budget", "AUC, pre-filing account-months"], rows,
                   f"{sep['clean_filers']:,} never-delinquent filers. The AUC column ranks pre-filing account-months with no 60+ DPD in their window against all non-filer account-months."))
    o.append(f"<p>At the shipped thresholds, both alerts together ({pct(sep['combined_alert_rate'])} of accounts a month) flag {pct(fb['shipped_both_alerts']['all_filers'])} of all filers before they file, "
             f"against {pct(fb['shipped_delinquency_alert']['all_filers'])} for the delinquency alert alone.</p>")

    o.append('<h2 id="reasons">Reason codes</h2>')
    o.append("<p>Reasons are the feature groups that add the most risk relative to the average account, taken straight from the scoring tables, at most four, one per group. "
             "This follows a method described in the official commentary to Regulation B for scored decisions (12 CFR 1002.9(b)(2), comment 5). When a value is absent and its absence adds risk, the reason says so.</p>")
    for n in ("delinquency", "bankruptcy"):
        r = v["reason_codes"][n]
        o.append(f"<h3>{n.capitalize()} alerts ({r['alerted_sampled']:,} sampled)</h3>")
        o.append(table(["Reason statement", "Share of alerts citing it", "Share where it is the first reason"],
                       [[k, pct(s), pct(r["top_first_reason"].get(k, 0.0))] for k, s in r["share_of_alerts_citing"].items()],
                       f"Average {r['mean_reasons']:.1f} reasons per alert; {pct(r['share_with_no_reason'])} of alerts have none."))

    o.append('<h2 id="actions">Action recommendation</h2>')
    o.append(f"<p>{pol['n_decisions']:,} accounts that tripped the legacy trigger were assigned at random to one of five arms. Value is {escape(pol['value_definition'])}, over {pol['horizon_months']} months. "
             "Each account was then assigned an action by models fitted without it, and each policy was valued on the accounts whose random arm matched. Because assignment was random with known probability, the estimates do not depend on the value models being right.</p>")
    o.append(table(["Arm", "Accounts", "Mean value", "Std. error", "Credit loss", "Revenue", "Any loss"],
                   [[ARM_LABEL[a], f"{r['n']:,}", usd(r["mean_value"]), usd(r["se"]), usd(r["loss"]), usd(r["revenue"]), pct(r["bad_rate"])] for a, r in pol["arms"].items()],
                   "Average outcome by randomized arm. Every blanket treatment is compared with no action in the next table."))
    keys = list(pol["policies"])
    o.append(C.bar_chart([POLICY_LABEL[k] for k in keys], [pol["policies"][k]["gain_vs_no_action"] for k in keys], "policy gain per pilot account versus no action",
                         fmt=lambda x: f"{'-' if x < 0 else ''}${abs(x):,.0f}", ci=[tuple(pol["policies"][k]["gain_ci95"]) for k in keys], highlight=len(keys) - 1, label_w=210))
    o.append(table(["Policy", "Value per account", "Gain vs no action", "95% interval", "P(gain ≤ 0)", "Share treated"],
                   [[POLICY_LABEL[k], usd(p["value"]), usd(p["gain_vs_no_action"]), f"{usd(p['gain_ci95'][0])} to {usd(p['gain_ci95'][1])}", num(p["p_gain_le_0"]), pct(1 - p["action_mix"]["none"])] for k, p in pol["policies"].items()],
                   "Bootstrap intervals over accounts, paired with the no-action policy."))
    if inconclusive:
        o.append(f"<p><b>Reading.</b> The Forewarn policy has the highest estimated value, but its interval includes zero (probability the gain is zero or negative: {fp['p_gain_le_0']:.2f}). "
                 "On a pilot of this size the recommendation layer is not shown to beat doing nothing. Blanket hardship plans and blanket limit decreases are shown to lose money.</p>")
    else:
        o.append(f"<p><b>Reading.</b> The Forewarn policy beats no action with an interval that excludes zero (probability the gain is zero or negative: {fp['p_gain_le_0']:.2f}).</p>")
    sr = pol["shipped_rule"]
    o.append("<h3>The rule the scorer ships</h3>"
             f"<p>The scorer recommends an action only for accounts that are alerted by a risk score and inside the pilot's population. That narrower rule was valued on the {sr['n_decisions']:,} pilot decisions made in months "
             f"{sr['decision_months'][0]} to {sr['decision_months'][1]}, after the risk models' training window, so the alerts are out of sample for every arm. "
             f"{pct(sr['share_alerted'])} of those accounts were alerted and the rule acts on {pct(sr['share_treated'])}.</p>")
    un = sr["ungated_policy_same_accounts"]
    o.append(table(["Policy, same accounts", "Gain vs no action", "95% interval", "P(gain ≤ 0)", "Share treated"],
                   [["Shipped rule (alert and pilot support required)", usd(sr["gain_vs_no_action"]), f"{usd(sr['gain_ci95'][0])} to {usd(sr['gain_ci95'][1])}", num(sr["p_gain_le_0"]), pct(sr["share_treated"])],
                    ["Value models applied to every account", usd(un["gain_vs_no_action"]), f"{usd(un['gain_ci95'][0])} to {usd(un['gain_ci95'][1])}", "", ""]]))
    o.append("<p>" + ("The shipped rule's interval includes zero as well. " if sr["gain_ci95"][0] <= 0 else "The shipped rule's interval excludes zero. ")
             + "It rests on fewer decisions than the table above, so its interval is wider.</p>")
    ba = pol["forewarn_policy_by_action"]
    o.append("<h3>Where the value models recommend each action</h3>")
    o.append(table(["Recommended action", "Share of pilot accounts", "Value when the action was given", "Value under no action", "Difference", "Std. error", "Accounts compared"],
                   [[ARM_LABEL[a], pct(r["share_of_pilot"]), usd(r["value_if_treated"]), usd(r["value_if_no_action"]), usd(r["difference"]), usd(r["difference_se"]), f"{r['n_treated_as_recommended']:,} vs {r['n_control']:,}"] for a, r in ba.items()],
                   "Accounts the value models would treat, split by the arm they were randomly given. A difference smaller than about twice its standard error is not evidence either way."))
    worse = [ARM_LABEL[a].lower() for a, r in ba.items() if r["difference"] < 0]
    if worse:
        clear = [ARM_LABEL[a].lower() for a, r in ba.items() if r["difference"] < -2 * r["difference_se"]]
        o.append(f"<p>Where a {' or a '.join(worse)} was recommended, accounts that received it did worse than those left alone"
                 + (". " if clear else ", though within the noise. ") + "This pilot gives no support for that arm's targeting.</p>")
    aa = v["actions_on_alerted"]
    o.append("<h3>What alerted accounts receive in the out-of-time window</h3>")
    o.append(table(["Outcome", "Share of alerted accounts"],
                   [[ARM_LABEL.get(k, k), pct(s)] for k, s in aa["mix"].items()] + [["— of which outside the pilot's population (no recommendation given)", pct(aa["notes"].get("outside_pilot_support", 0.0))],
                                                                                    ["— adverse-action notice required (limit decrease)", pct(aa["adverse_action_notice_share"])]]))

    o.append('<h2 id="fair">Fair lending tests</h2>')
    fl = v["fair_lending"]
    o.append(f"<p>{escape(fl['note'])} Protected attributes are joined for testing only and are never model inputs. "
             "The tests compare how often each group is alerted, whether the score is equally accurate and equally calibrated for each group, and whether any feature stands in for group membership.</p>")
    for attr, title in (("group_a", "Group A (correlated with income)"), ("group_b", "Group B (negative control)")):
        a = fl["attributes"][attr]
        o.append(f"<h3>{title}: {pct(a['share_protected'])} of accounts</h3>")
        rows = []
        for n in ("delinquency", "bankruptcy"):
            m = a["models"][n]
            rows.append([n.capitalize(), pct(m["alert_rate_protected"], 2), pct(m["alert_rate_reference"], 2), num(m["adverse_rate_ratio"], 2), num(m["favorable_outcome_ratio_AIR"], 3),
                         f"{m['standardized_mean_difference_logit']:+.2f}", f"{num(m['auc_protected'])} / {num(m['auc_reference'])}",
                         f"{num(m['observed_over_predicted_protected'], 2)} / {num(m['observed_over_predicted_reference'], 2)}",
                         f"{pct(m['false_positive_rate_protected'], 2)} / {pct(m['false_positive_rate_reference'], 2)}"])
        o.append(table(["Score", "Alert rate, protected", "Alert rate, reference", "Alert-rate ratio", "Not-alerted ratio (AIR)", "Score gap (SMD)", "AUC prot / ref", "Obs/pred prot / ref", "False-positive rate prot / ref"], rows))
        o.append(table(["Feature", "AUC predicting group", "Missing, protected", "Missing, reference"],
                       [[p["feature"], num(p["auc_predicting_group"]), pct(p["missing_rate_protected"]), pct(p["missing_rate_reference"])] for p in a["proxy_screen_top"]],
                       f"Proxy screen, six strongest. 0.5 means no information about group membership. The scores themselves predict group membership with AUC {num(a['score_auc_predicting_group']['delinquency'])} (delinquency) and {num(a['score_auc_predicting_group']['bankruptcy'])} (bankruptcy)."))
    lda = fl["lda_search"]
    o.append("<h3>Search for a less discriminatory alternative</h3>")
    o.append(table(["Delinquency model refitted without", "AUC", "Events in top 5%", "Group A alert-rate ratio"],
                   [[r["dropped"], num(r["auc"], 4), pct(r["capture_top5"]), num(r["adverse_rate_ratio"], 3)] for r in lda["rows"]],
                   f"Each row refits the model on a {pct(lda['training_sample'], 0)} sample of training rows without one of the three strongest proxy features."))
    o.append('<p class="small">Regulatory context, United States: the CFPB\'s 2026 amendments to Regulation B (effective July 21, 2026) state that ECOA does not authorise disparate-impact claims; disparate treatment, including the use of proxies designed to stand in for a prohibited basis, remains prohibited, and state law may still reach disparate impact. '
             'These tests are kept because a bank\'s compliance function will ask for them either way. This is not legal advice.</p>')

    o.append('<h2 id="drift">Population stability</h2>')
    dr = v["drift"]
    top = dr["features"][:10]
    o.append("<p>Population stability index of each input between the training window and the out-of-time window, using the reference bins shipped in the bundle. The scorer computes the same index on every batch it scores.</p>")
    o.append(C.bar_chart([r["feature"] for r in top], [r["psi_train_to_test"] for r in top], "PSI by feature, training to out-of-time", fmt=lambda x: f"{x:.3f}",
                         refs=[(0.10, "watch 0.10"), (0.25, "alert 0.25")]))
    rows = []
    for n in ("delinquency", "bankruptcy"):
        s = dr["scores"][n]
        rows.append([f"{n.capitalize()} score", num(s["psi_train_to_test"]), *[num(m["psi"]) for m in s["by_month"]]])
    o.append(table(["Score PSI", "Whole window", *[f"m{m['month']}" for m in dr["scores"]["delinquency"]["by_month"]]], rows,
                   f"The simulator raises job-loss hazard by {v['simulator'].get('downturn_multiplier', 1):g}x from month {v['simulator'].get('downturn_start', '?')}; the observed-over-predicted ratios above show the effect on calibration."))

    sm = dr.get("single_month", [])
    if sm:
        cal = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        o.append("<h3>One month at a time</h3><p>A monthly batch run compares a single month with the pooled training reference. Individual inputs then move for reasons that are not model decay, mostly seasonal spending. The scores move far less. The scorer therefore reports score drift and input drift separately, and by default only score drift fails a run.</p>")
        o.append(table(["Month", "Calendar", "Largest input PSI", "Input", "Inputs ≥ 0.10", "Inputs ≥ 0.25", "Delinquency score PSI", "Bankruptcy score PSI"],
                       [[r["month"], cal[r["calendar_month"] - 1], num(r["max_feature_psi"]), r["max_feature"], r["features_at_watch"], r["features_at_alert"],
                         num(r["score_psi"]["delinquency"]), num(r["score_psi"]["bankruptcy"])] for r in sm]))

    o.append('<h2 id="impl">Implementation verification</h2>')
    if par:
        o.append(f"<p>The Go scorer replayed {par['cases']} reference cases produced by the Python implementation: {par['passed']} matched on every score, alert, reason and recommended action. "
                 f"Largest absolute difference: {par['max_abs_diff_logit']:.1e} on a logit, {par['max_abs_diff_probability']:.1e} on a probability, {par['max_abs_diff_gain']:.1e} on an estimated gain. "
                 f"Bundle digest checked: <code>{par['bundle_sha256'][:16]}…</code>. Anyone holding the bundle and the cases can repeat this with <code>forewarn verify</code>.</p>")
    else:
        o.append("<p>No parity result found. Run <code>make verify</code> and regenerate this report.</p>")

    o.append('<h2 id="limits">Limitations</h2><ul>')
    for item in [
        "All data are simulated. Lift, lead time and intervention effects are properties of the simulator's assumptions and will differ on a real book.",
        "The simulated bankruptcy rate is set above what a typical card portfolio sees so that the test window contains enough filings to measure anything.",
        "The champion stand-in has no credit-bureau data. Measured lift over a bank's actual behaviour score will be smaller.",
        "The models are additive. Interactions between features are not captured; the benchmark comparison bounds what that costs here, not on other data.",
        "Action recommendations rest on a randomized pilot. Without one on real accounts, the recommendation layer cannot be fitted or validated, and historical collector decisions are not a substitute.",
        "Recommendations are only made inside the population the pilot enrolled. Alerted accounts outside it get no recommendation.",
        "Intervention value counts interest, interchange and cash-advance fees and excludes penalty fees. A bank that values these differently will get a different policy.",
        "Calibration is fitted on one window. The report shows it moving under a simulated downturn; a production deployment needs a recalibration trigger tied to the observed-over-predicted ratio.",
        "Outcome windows are six months for the scores and eighteen for action value. Effects that appear later are not measured.",
    ]:
        o.append(f"<li>{escape(item)}</li>")
    o.append("</ul>")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Forewarn validation report</title><style>{CSS}</style></head><body><main>{"".join(o)}</main></body></html>'


def render_model_card(v: dict) -> str:
    md, mb = v["models"]["delinquency"], v["models"]["bankruptcy"]
    lt = v["lead_time"]["models"]["forewarn_delinquency"]
    sep, pol = v["separate_target"], v["policy"]["policies"]["forewarn_policy"]
    allowed = [f for f in FEATURES if f.status == "allowed"]
    lines = [
        "# Forewarn model card",
        "",
        f"Bundle `{v['bundle_sha256']}` · data source: **{v['data_source']}**",
        "",
        "> Trained and evaluated on simulated customers. Not for decisions about real customers.",
        "",
        "## What it does",
        "",
        "| Output | Definition |",
        "|---|---|",
        "| Delinquency score | Probability an account that is current today is 60+ days past due within 6 months |",
        "| Bankruptcy score | Probability an account that is current today files for bankruptcy within 6 months |",
        "| Reasons | Up to four feature groups adding the most risk relative to the average account, exact from the scoring tables |",
        "| Recommended action | Payment-date change, hardship plan, limit decrease, outreach call, or none |",
        "",
        "## Inputs",
        "",
        f"{len(allowed)} features computed from the bank's own deposit transactions, card transactions and card statements. No bureau file, no data from another institution.",
        "",
        "| Feature | Source | Direction | Reason statement |",
        "|---|---|---|---|",
    ]
    arrow = {1: "risk rises with value", -1: "risk falls with value", 0: "unconstrained"}
    for f in allowed:
        lines.append(f"| `{f.name}` | {f.source} | {arrow[f.monotone]} | {f.reason_text} |")
    lines += [
        "",
        "Excluded by policy: `legal_services_3m` (restricted, merchant-category behaviour) and `gov_benefit_flag_3m` (prohibited basis under Regulation B).",
        "",
        "## Model form",
        "",
        "Gradient-boosted depth-one trees with monotone constraints, folded into one lookup table per feature. The score is an intercept plus one table value per feature, passed through a logistic calibration.",
        "",
        "## Out-of-time performance",
        "",
        "| | Delinquency | Bankruptcy |",
        "|---|---|---|",
        f"| AUC | {md['comparators']['forewarn']['auc']:.3f} | {mb['comparators']['forewarn']['auc']:.3f} |",
        f"| 95% interval | {md['comparators']['forewarn']['auc_ci95'][0]:.3f} to {md['comparators']['forewarn']['auc_ci95'][1]:.3f} | {mb['comparators']['forewarn']['auc_ci95'][0]:.3f} to {mb['comparators']['forewarn']['auc_ci95'][1]:.3f} |",
        f"| Champion stand-in AUC | {md['comparators']['champion']['auc']:.3f} | {mb['comparators']['champion']['auc']:.3f} |",
        f"| Unconstrained benchmark AUC | {md['comparators']['benchmark_unconstrained']['auc']:.3f} | {mb['comparators']['benchmark_unconstrained']['auc']:.3f} |",
        f"| Observed / predicted | {md['observed_over_predicted_test']:.2f} | {mb['observed_over_predicted_test']:.2f} |",
        f"| Alert threshold (probability) | {md['alert_threshold']:.4f} | {mb['alert_threshold']:.4f} |",
        "",
        f"- Alerted at least 75 days before the first missed payment: {pct(lt['share_with_lead_at_least']['75'])} of accounts that went on to 60+ days past due after a year with no missed payment.",
        f"- Filers never 60 days past due in the 12 months before filing: {pct(sep['share_never_60dpd_in_prior_12m'])}. Flagged before filing at a 5% budget by the bankruptcy score: {pct(sep['flagged_before_filing']['bankruptcy_score_top5pct']['clean_filers'])}; by the delinquency score: {pct(sep['flagged_before_filing']['delinquency_score_top5pct']['clean_filers'])}.",
        f"- Action policy gain over no action, cross-fitted on the randomized pilot: {usd(pol['gain_vs_no_action'])} per account (95% interval {usd(pol['gain_ci95'][0])} to {usd(pol['gain_ci95'][1])}). The rule the scorer ships: {usd(v['policy']['shipped_rule']['gain_vs_no_action'])} ({usd(v['policy']['shipped_rule']['gain_ci95'][0])} to {usd(v['policy']['shipped_rule']['gain_ci95'][1])}).",
        "",
        "## Limits of use",
        "",
        "- Current, open revolving card accounts only. Not for origination, pricing or collections.",
        "- A limit decrease on a current account is adverse action under Regulation B; the scorer marks those decisions and supplies the reasons.",
        "- Recommendations are advisory. A person or a governed decision process acts on them.",
        "- Recalibrate when observed over predicted leaves the band the bank sets. Investigate when a score's PSI reaches 0.25; single inputs cross that level seasonally without the scores moving.",
        "- The action recommendation has no demonstrated benefit on the reference run and should be treated as experimental.",
        "",
        "Full evidence: `validation_report.html`.",
        "",
    ]
    return "\n".join(lines)


def render_results(v: dict) -> str:
    """Headline numbers as Markdown, one block per product claim."""
    md, mb = v["models"]["delinquency"], v["models"]["bankruptcy"]
    cd, cb = md["comparators"], mb["comparators"]
    lt = v["lead_time"]
    fw, ch = lt["models"]["forewarn_delinquency"], lt["models"]["champion_delinquency"]
    sep, fb, au = v["separate_target"], v["separate_target"]["flagged_before_filing"], v["separate_target"]["auc_account_months"]
    rm = lt["recent_miss"]["forewarn_delinquency"]
    sr = v["policy"]["shipped_rule"]
    pol, ab = v["policy"], v["alert_burden"]
    par = v.get("implementation_verification")
    fa = v["fair_lending"]["attributes"]["group_a"]["models"]
    L = [
        "# Results of the last run",
        "",
        f"Bundle `{v['bundle_sha256']}`. Data source: **{v['data_source']}**. "
        f"{v['rows']['train']:,} training, {v['rows']['valid']:,} validation and {v['rows']['test']:,} out-of-time account-months "
        f"({v['customers_test']:,} customers in the out-of-time window).",
        "",
        "> Simulated customers. These numbers show the pipeline works and what it measures. They are not evidence about a real portfolio.",
        "",
        "## 1. Early warning from the bank's own transactions",
        "",
        "| Out-of-time, 60+ days past due within 6 months | AUC | 95% interval | Events in top 5% | 12-month loss $ in top 5% (snapshots with 12 months of outcome) |",
        "|---|---|---|---|---|",
    ]
    for key in ("forewarn", "card_only", "champion", "benchmark_unconstrained"):
        c = cd[key]
        L.append(f"| {LABELS[key]} | {c['auc']:.3f} | {c['auc_ci95'][0]:.3f} to {c['auc_ci95'][1]:.3f} | {pct(c['top_5pct']['event_capture'])} | {pct(c['top_5pct'].get('amount_capture'))} |")
    L += [
        "",
        f"Lead time, measured on {lt['events']:,} accounts whose missed payment led to 60+ days past due and who had missed nothing in the {lt['clean_months']} months before:",
        "",
        "| Alerted at least | " + " | ".join(f"{b} days" for b in lt["buckets_days"]) + " |",
        "|---|" + "---|" * len(lt["buckets_days"]),
        "| Forewarn | " + " | ".join(pct(fw["share_with_lead_at_least"][str(b)], 0) for b in lt["buckets_days"]) + " |",
        "| Champion stand-in, same alert rate | " + " | ".join(pct(ch["share_with_lead_at_least"][str(b)], 0) for b in lt["buckets_days"]) + " |",
        "",
        f"A further {lt['events_with_recent_miss']:,} accounts had a missed payment in the prior {lt['clean_months']} months; {pct(rm['share_with_lead_at_least']['75'], 0)} of those were alerted 75+ days early, "
        "but a score that has already seen a miss is not giving early warning, so they are excluded above.",
        "",
        f"Alert burden: {ab['alerts_per_1000_accounts_per_month']:.0f} accounts per 1,000 per month; {pct(ab['precision_bad_within_6m'])} of them reach 60+ days past due or file within 6 months; "
        f"alerts hold {pct(ab['share_of_12m_loss_dollars_in_alerts'])} of 12-month loss dollars (snapshots with 12 months of outcome).",
        "",
        "## 2. Recommend the action",
        "",
        f"{pol['n_decisions']:,} pilot accounts randomized across five arms; value over {pol['horizon_months']} months; policies cross-fitted and valued by inverse propensity weighting.",
        "",
        "| Policy | Gain per account vs no action | 95% interval | Share treated |",
        "|---|---|---|---|",
    ]
    for k, p_ in pol["policies"].items():
        if k == "no_action":
            continue
        L.append(f"| {POLICY_LABEL[k]} | {usd(p_['gain_vs_no_action'])} | {usd(p_['gain_ci95'][0])} to {usd(p_['gain_ci95'][1])} | {pct(1 - p_['action_mix']['none'])} |")
    L += [
        "",
        f"The rule the scorer ships acts only on alerted accounts inside pilot support. Valued on the {sr['n_decisions']:,} pilot decisions made after the risk models' training window: "
        f"{usd(sr['gain_vs_no_action'])} per account ({usd(sr['gain_ci95'][0])} to {usd(sr['gain_ci95'][1])}), acting on {pct(sr['share_treated'])} of them.",
    ]
    L += [
        "",
        "## 3. Bankruptcy as a separate target",
        "",
        f"{sep['filers']:,} filers; {pct(sep['share_never_60dpd_in_prior_12m'])} were never 60 days past due in the 12 months before filing. Counted per filer: flagged in any of the six snapshots before filing.",
        "",
        "| Score | Never-delinquent filers flagged, 2% budget | Never-delinquent filers flagged, 5% budget | All filers flagged, 5% budget |",
        "|---|---|---|---|",
        f"| Bankruptcy score | {pct(fb['bankruptcy_score_top2pct']['clean_filers'])} | {pct(fb['bankruptcy_score_top5pct']['clean_filers'])} | {pct(fb['bankruptcy_score_top5pct']['all_filers'])} |",
        f"| Forewarn delinquency score | {pct(fb['delinquency_score_top2pct']['clean_filers'])} | {pct(fb['delinquency_score_top5pct']['clean_filers'])} | {pct(fb['delinquency_score_top5pct']['all_filers'])} |",
        f"| Champion stand-in | {pct(fb['champion_delinquency_top2pct']['clean_filers'])} | {pct(fb['champion_delinquency_top5pct']['clean_filers'])} | {pct(fb['champion_delinquency_top5pct']['all_filers'])} |",
        "",
        f"Bankruptcy model overall: AUC {cb['forewarn']['auc']:.3f} ({cb['forewarn']['auc_ci95'][0]:.3f} to {cb['forewarn']['auc_ci95'][1]:.3f}); champion stand-in {cb['champion']['auc']:.3f}; unconstrained benchmark {cb['benchmark_unconstrained']['auc']:.3f}.",
        "",
        "## 4. Governance evidence",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Cost of the readable model form | {cd['benchmark_unconstrained']['auc'] - cd['forewarn']['auc']:.3f} AUC (delinquency), {cb['benchmark_unconstrained']['auc'] - cb['forewarn']['auc']:.3f} (bankruptcy) against unconstrained boosting |",
        f"| Calibration, observed over predicted | {md['observed_over_predicted_test']:.2f} (delinquency), {mb['observed_over_predicted_test']:.2f} (bankruptcy) |",
        f"| Monotone tables | {sum(1 for t in md['terms'] + mb['terms'] if t['monotone_ok'])} constrained terms checked, {sum(1 for t in md['terms'] + mb['terms'] if t['monotone_ok'] is False)} violations |",
        f"| Reasons per alert | {v['reason_codes']['delinquency']['mean_reasons']:.1f} on average; {pct(v['reason_codes']['delinquency']['share_with_no_reason'])} of alerts with none |",
        f"| Group A alert-rate ratio | {fa['delinquency']['adverse_rate_ratio']:.2f} (delinquency), {fa['bankruptcy']['adverse_rate_ratio']:.2f} (bankruptcy) |",
        f"| Largest feature PSI, training to out-of-time | {v['drift']['features'][0]['psi_train_to_test']:.3f} ({v['drift']['features'][0]['feature']}) |",
        f"| Restricted feature, AUC it would add | {cd['with_restricted_feature']['auc'] - cd['forewarn']['auc']:+.4f} (delinquency), {cb['with_restricted_feature']['auc'] - cb['forewarn']['auc']:+.4f} (bankruptcy) |",
        "",
        "## 5. Runs inside the bank's account",
        "",
        "| Item | Result |",
        "|---|---|",
        f"| Bundle size | {v['export']['bytes'] / 1024:.0f} KiB, {v['export']['features']} input features |",
    ]
    if par:
        L.append(f"| Go scorer against Python reference | {par['passed']} of {par['cases']} cases identical; largest logit difference {par['max_abs_diff_logit']:.1e} |")
    L += ["| Scorer dependencies | Go standard library only |", ""]
    return "\n".join(L)


def render(store: Store) -> list[Path]:
    v = store.read_json("metrics/validation.json")
    try:
        par = store.read_json("metrics/parity.json")
        # a parity result for another bundle says nothing about this one
        v["implementation_verification"] = par if par.get("bundle_sha256") == v["bundle_sha256"] else None
    except FileNotFoundError:
        v["implementation_verification"] = None
    out = store.artifacts / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "validation_report.html").write_text(render_html(v))
    (out / "model_card.md").write_text(render_model_card(v))
    (out / "results.md").write_text(render_results(v))
    return [out / "validation_report.html", out / "model_card.md", out / "results.md"]
