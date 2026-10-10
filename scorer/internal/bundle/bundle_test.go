package bundle_test

import (
	"strings"
	"testing"

	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
	"github.com/udaykishore-resu/forewarn/scorer/internal/testutil"
)

func TestValidBundleLoads(t *testing.T) {
	b, err := testutil.Bundle(t)
	if err != nil {
		t.Fatal(err)
	}
	if len(b.SHA256) != 64 {
		t.Fatalf("digest not set: %q", b.SHA256)
	}
}

// Each case breaks one promise the documentation makes about a bundle.
func TestValidationRejects(t *testing.T) {
	cases := []struct {
		name, old, new, want string
	}{
		{"prohibited feature", `"reason_code": "D06", "reason_text": "Deposit balance low relative to monthly spending", "missing_text": "No deposit account history with us", "status": "allowed"`,
			`"reason_code": "D06", "reason_text": "x", "missing_text": "x", "status": "prohibited"`, "may not be used"},
		{"restricted feature without sign-off", `"missing_text": "", "status": "allowed"`, `"missing_text": "", "status": "restricted"`, "restricted"},
		{"table against declared direction", `"values": [-0.5, 0.25, 1.5]`, `"values": [-0.5, 1.5, 0.25]`, "not monotone"},
		{"thresholds out of order", `"thresholds": [0.3, 0.8], "values": [-0.5, 0.25, 1.5]`, `"thresholds": [0.8, 0.3], "values": [-0.5, 0.25, 1.5]`, "strictly increasing"},
		{"wrong number of values", `"values": [-0.1, 0.75]`, `"values": [-0.1]`, "values for"},
		{"term on unknown feature", `{"feature": "util_change", "thresholds": [0.1]`, `{"feature": "zip_code", "thresholds": [0.1]`, "unknown feature"},
		{"threshold logit and probability disagree", `"threshold_probability": 0.5, "threshold_logit": 0.0, "share_alerted_on_validation": 0.05`,
			`"threshold_probability": 0.5, "threshold_logit": 1.0, "share_alerted_on_validation": 0.05`, "disagree"},
		{"first arm is not none", `"arms": ["none", "hardship_plan", "limit_decrease"]`, `"arms": ["hardship_plan", "none", "limit_decrease"]`, "must start with"},
		{"unsupported operator", `"op": ">=", "value": 200.0`, `"op": "==", "value": 200.0`, "unsupported operator"},
		{"schema version", `"schema_version": 1`, `"schema_version": 2`, "not supported"},
		{"calibration slope reverses the score", `"calibration": {"a": 0.5, "b": 2.0}`, `"calibration": {"a": 0.5, "b": -2.0}`, "slope must be positive"},
		{"drift reference shape", `"shares": [0.5, 0.3, 0.2, 0.0]`, `"shares": [0.5, 0.5]`, "shares"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if !strings.Contains(testutil.Tiny, c.old) {
				t.Fatalf("test is stale: fragment not found in fixture")
			}
			_, err := testutil.Bundle(t, c.old, c.new)
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want error containing %q, got %v", c.want, err)
			}
		})
	}
}

func TestRestrictedAllowedWithSignOff(t *testing.T) {
	raw := strings.Replace(testutil.Tiny, `"missing_text": "", "status": "allowed"`, `"missing_text": "", "status": "restricted"`, 1)
	if _, err := bundle.Parse([]byte(raw), bundle.Options{AllowRestricted: true}); err != nil {
		t.Fatal(err)
	}
}

func TestDigestPinning(t *testing.T) {
	b, _ := testutil.Bundle(t)
	if _, err := bundle.Parse([]byte(testutil.Tiny), bundle.Options{ExpectSHA256: b.SHA256}); err != nil {
		t.Fatalf("matching digest rejected: %v", err)
	}
	// one edited table value must fail closed
	edited := strings.Replace(testutil.Tiny, `"intercept": -3.0`, `"intercept": -3.1`, 1)
	if _, err := bundle.Parse([]byte(edited), bundle.Options{ExpectSHA256: b.SHA256}); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("edited bundle accepted: %v", err)
	}
}
