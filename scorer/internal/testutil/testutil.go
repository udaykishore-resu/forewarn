// Package testutil holds a small hand-written bundle whose arithmetic can be
// checked by eye. It is used only by tests.
package testutil

import (
	"strings"
	"testing"

	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
)

// Tiny is a valid bundle with three features.
//
//	delinquency = -3 + f(utilization) + f(util_change) + f(buffer_days)
//	bankruptcy  = -5 + f(utilization)
const Tiny = `{
 "schema_version": 1, "name": "forewarn", "version": "test", "created_at": "2026-01-01T00:00:00+00:00", "data_source": "synthetic",
 "training": {},
 "features": [
  {"name": "utilization", "group": "utilization", "source": "card", "monotone": 1, "reason_code": "C02", "reason_text": "Balance high relative to credit limit", "missing_text": "Insufficient recent card activity to evaluate", "status": "allowed"},
  {"name": "util_change", "group": "utilization", "source": "card", "monotone": 1, "reason_code": "C03", "reason_text": "Recent increase in balance", "missing_text": "Insufficient recent card activity to evaluate", "status": "allowed"},
  {"name": "buffer_days", "group": "buffer", "source": "deposit", "monotone": -1, "reason_code": "D06", "reason_text": "Deposit balance low relative to monthly spending", "missing_text": "No deposit account history with us", "status": "allowed"},
  {"name": "open_to_buy", "group": "exposure", "source": "card", "monotone": 0, "reason_code": "V02", "reason_text": "", "missing_text": "", "status": "allowed"}
 ],
 "models": {
  "delinquency": {"name": "delinquency", "link": "logit", "intercept": -3.0, "calibration": {"a": 0.0, "b": 1.0}, "terms": [
    {"feature": "utilization", "thresholds": [0.3, 0.8], "values": [-0.5, 0.25, 1.5], "missing": 0.0},
    {"feature": "util_change", "thresholds": [0.1], "values": [-0.1, 0.75], "missing": 0.0},
    {"feature": "buffer_days", "thresholds": [10.0, 40.0], "values": [2.0, 0.5, -1.0], "missing": 0.4}
  ]},
  "bankruptcy": {"name": "bankruptcy", "link": "logit", "intercept": -5.0, "calibration": {"a": 0.5, "b": 2.0}, "terms": [
    {"feature": "utilization", "thresholds": [0.8], "values": [-0.25, 2.0], "missing": 0.0}
  ]},
  "value_effects": {
   "none": {"name": "none", "link": "identity", "intercept": 10.0, "calibration": {"a": 0, "b": 1}, "terms": []},
   "hardship_plan": {"name": "hardship_plan", "link": "identity", "intercept": -40.0, "calibration": {"a": 0, "b": 1}, "terms": [
     {"feature": "buffer_days", "thresholds": [10.0], "values": [300.0, 0.0], "missing": 0.0}
   ]},
   "limit_decrease": {"name": "limit_decrease", "link": "identity", "intercept": -90.0, "calibration": {"a": 0, "b": 1}, "terms": [
     {"feature": "open_to_buy", "thresholds": [1000.0], "values": [0.0, 500.0], "missing": 0.0}
   ]}
  }
 },
 "alerts": {
  "delinquency": {"threshold_probability": 0.5, "threshold_logit": 0.0, "share_alerted_on_validation": 0.05},
  "bankruptcy": {"threshold_probability": 0.5, "threshold_logit": 0.0, "share_alerted_on_validation": 0.02}
 },
 "reasons": {"max_reasons": 4, "min_points": 0.02},
 "policy": {
  "arms": ["none", "hardship_plan", "limit_decrease"],
  "eligibility": {"limit_decrease": [{"feature": "open_to_buy", "op": ">=", "value": 200.0}]},
  "pilot_support": [[{"feature": "utilization", "op": ">=", "value": 0.5}]],
  "min_gain": 50.0, "adverse_actions": ["limit_decrease"], "value_horizon_months": 18
 },
 "drift_reference": {
  "features": {"utilization": {"edges": [0.3, 0.8], "shares": [0.5, 0.3, 0.2, 0.0]}},
  "scores": {"delinquency": {"edges": [-2.0, 0.0], "shares": [0.6, 0.3, 0.1, 0.0]}}
 }
}`

// Bundle parses Tiny, optionally after string replacements (old, new pairs).
func Bundle(t testing.TB, replace ...string) (*bundle.Bundle, error) {
	t.Helper()
	return bundle.Parse([]byte(strings.NewReplacer(replace...).Replace(Tiny)), bundle.Options{})
}

// F builds a feature map; a nil pointer means missing.
func F(kv map[string]float64) map[string]*float64 {
	out := map[string]*float64{}
	for k, v := range kv {
		v := v
		out[k] = &v
	}
	return out
}
