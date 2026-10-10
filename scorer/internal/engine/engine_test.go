package engine_test

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
	"github.com/udaykishore-resu/forewarn/scorer/internal/engine"
	"github.com/udaykishore-resu/forewarn/scorer/internal/testutil"
)

func newEngine(t *testing.T) *engine.Engine {
	t.Helper()
	b, err := testutil.Bundle(t)
	if err != nil {
		t.Fatal(err)
	}
	e, err := engine.New(b)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func decide(t *testing.T, e *engine.Engine, kv map[string]float64) engine.Decision {
	t.Helper()
	in, err := e.Vector(testutil.F(kv))
	if err != nil {
		t.Fatal(err)
	}
	return e.Decide(in)
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestScoreArithmetic(t *testing.T) {
	e := newEngine(t)
	// -3 + 1.5 (util 0.9) + 0.75 (change 0.2) + 2.0 (buffer 5) = 1.25
	d := decide(t, e, map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 5, "open_to_buy": 100})
	if got := d.Scores["delinquency"].Logit; !near(got, 1.25) {
		t.Fatalf("delinquency logit = %v, want 1.25", got)
	}
	// bankruptcy: raw = -5 + 2 = -3; z = 0.5 + 2*(-3) = -5.5
	if got := d.Scores["bankruptcy"].Logit; !near(got, -5.5) {
		t.Fatalf("bankruptcy logit = %v, want -5.5", got)
	}
	if !d.Scores["delinquency"].Alert || d.Scores["bankruptcy"].Alert || !d.Alert || d.Driver != "delinquency" {
		t.Fatalf("alerts wrong: %+v", d)
	}
	if p := d.Scores["delinquency"].Probability; !near(p, 1/(1+math.Exp(-1.25))) {
		t.Fatalf("probability = %v", p)
	}
}

// A value equal to a threshold belongs to the lower bin: bins are (lo, hi].
func TestBinBoundaries(t *testing.T) {
	e := newEngine(t)
	base := map[string]float64{"util_change": 0.0, "buffer_days": 100, "open_to_buy": 0}
	for _, c := range []struct{ util, want float64 }{
		{0.3, -0.5}, {0.3000001, 0.25}, {0.8, 0.25}, {0.81, 1.5}, {-5, -0.5}, {99, 1.5},
	} {
		base["utilization"] = c.util
		got := decide(t, e, base).Scores["delinquency"].Logit
		want := -3 + c.want - 0.1 - 1.0
		if !near(got, want) {
			t.Errorf("utilization %v: logit %v, want %v", c.util, got, want)
		}
	}
}

func TestMissingValues(t *testing.T) {
	e := newEngine(t)
	// buffer_days absent -> its missing contribution 0.4; explicit null behaves the same
	d := decide(t, e, map[string]float64{"utilization": 0.9, "util_change": 0.2, "open_to_buy": 100})
	if got := d.Scores["delinquency"].Logit; !near(got, -3+1.5+0.75+0.4) {
		t.Fatalf("logit with missing buffer = %v", got)
	}
	f := testutil.F(map[string]float64{"utilization": 0.9, "util_change": 0.2, "open_to_buy": 100})
	f["buffer_days"] = nil
	in, err := e.Vector(f)
	if err != nil {
		t.Fatal(err)
	}
	if got := e.Decide(in).Scores["delinquency"].Logit; !near(got, -3+1.5+0.75+0.4) {
		t.Fatalf("logit with null buffer = %v", got)
	}
	rs := d.Scores["delinquency"].Reasons
	if len(rs) != 2 || rs[1].Code != "D06M" || rs[1].Text != "No deposit account history with us" {
		t.Fatalf("missing-value reason wrong: %+v", rs)
	}
}

func TestRequiredFeatures(t *testing.T) {
	b, err := testutil.Bundle(t, `"reason_text": "Balance high relative to credit limit", "missing_text": "Insufficient recent card activity to evaluate", "status": "allowed"`,
		`"reason_text": "Balance high relative to credit limit", "missing_text": "Insufficient recent card activity to evaluate", "status": "allowed", "required": true`)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := engine.New(b)
	// utilization was never missing in training: absent or null is refused, not scored
	if _, err := e.Vector(testutil.F(map[string]float64{"buffer_days": 5})); err == nil || !strings.Contains(err.Error(), "required") {
		t.Fatalf("absent required feature accepted: %v", err)
	}
	f := testutil.F(map[string]float64{"buffer_days": 5})
	f["utilization"] = nil
	if _, err := e.Vector(f); err == nil {
		t.Fatal("null required feature accepted")
	}
	// buffer_days may be missing
	if _, err := e.Vector(testutil.F(map[string]float64{"utilization": 0.5})); err != nil {
		t.Fatalf("optional feature wrongly required: %v", err)
	}
}

func TestInputValidation(t *testing.T) {
	e := newEngine(t)
	if _, err := e.Vector(testutil.F(map[string]float64{"zip_code": 20170})); err == nil {
		t.Fatal("unknown feature accepted")
	}
	if _, err := e.Vector(testutil.F(map[string]float64{"utilization": math.Inf(1)})); err == nil {
		t.Fatal("infinite value accepted")
	}
}

// Reasons are grouped: utilization and util_change share a group, so they
// produce one reason carrying the sum of their points and the text of the
// larger contributor.
func TestReasons(t *testing.T) {
	e := newEngine(t)
	d := decide(t, e, map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 5, "open_to_buy": 100})
	rs := d.Scores["delinquency"].Reasons
	if len(rs) != 2 {
		t.Fatalf("want 2 reasons, got %+v", rs)
	}
	if rs[0].Code != "C02" || !near(rs[0].Points, 2.25) {
		t.Errorf("first reason = %+v, want C02 with 2.25 points", rs[0])
	}
	if rs[1].Code != "D06" || !near(rs[1].Points, 2.0) {
		t.Errorf("second reason = %+v, want D06 with 2.0 points", rs[1])
	}
	// a feature that lowers risk is never a reason
	d = decide(t, e, map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 90, "open_to_buy": 100})
	for _, r := range d.Scores["delinquency"].Reasons {
		if r.Feature == "buffer_days" {
			t.Errorf("risk-reducing feature cited as a reason: %+v", r)
		}
	}
}

func TestPolicy(t *testing.T) {
	e := newEngine(t)
	alerted := map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 5}

	// hardship gain = (-40 + 300) - 10 = 250; limit decrease gain = (-90 + 500) - 10 = 400 when eligible
	alerted["open_to_buy"] = 1500
	d := decide(t, e, alerted)
	if d.Action.Recommended != "limit_decrease" || !d.Action.AdverseActionNotice || !near(d.Action.EstimatedGain["limit_decrease"], 400) {
		t.Fatalf("want limit_decrease with gain 400 and a notice flag, got %+v", d.Action)
	}

	// not eligible for a limit decrease (open-to-buy below 200): falls back to the next best action
	alerted["open_to_buy"] = 150
	d = decide(t, e, alerted)
	if d.Action.Recommended != "hardship_plan" || d.Action.AdverseActionNotice || d.Action.Eligible["limit_decrease"] {
		t.Fatalf("want hardship_plan, got %+v", d.Action)
	}

	// alerted, but the best eligible gain (hardship: -40 - 10 = -50) is below the minimum: no action
	b2, err := testutil.Bundle(t, `"intercept": -3.0`, `"intercept": -2.0`)
	if err != nil {
		t.Fatal(err)
	}
	e2, _ := engine.New(b2)
	d = decide(t, e2, map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 20, "open_to_buy": 150})
	if !d.Alert || d.Action.Recommended != "none" || d.Action.Note != "no_action_clears_minimum_gain" {
		t.Fatalf("want an alert with no action, got alert=%v %+v", d.Alert, d.Action)
	}

	// not alerted: never a recommendation, whatever the estimated gain
	d = decide(t, e, map[string]float64{"utilization": 0.6, "util_change": 0.0, "buffer_days": 5, "open_to_buy": 1500})
	if d.Alert || d.Action.Recommended != "none" || d.Action.Note != "below_alert_threshold" {
		t.Fatalf("want below_alert_threshold, got alert=%v %+v", d.Alert, d.Action)
	}
}

func TestOutsidePilotSupport(t *testing.T) {
	// alert through bankruptcy with utilization forced below the support rule is impossible in
	// this fixture, so move the support rule instead
	b, err := testutil.Bundle(t, `{"feature": "utilization", "op": ">=", "value": 0.5}`, `{"feature": "utilization", "op": ">=", "value": 0.95}`)
	if err != nil {
		t.Fatal(err)
	}
	e, _ := engine.New(b)
	d := decide(t, e, map[string]float64{"utilization": 0.9, "util_change": 0.2, "buffer_days": 5, "open_to_buy": 1500})
	if !d.Alert || d.Action.Recommended != "none" || d.Action.Note != "outside_pilot_support" || d.Action.InPilotSupport {
		t.Fatalf("want outside_pilot_support, got %+v", d.Action)
	}
}

// TestParity replays the reference cases exported by the Python pipeline, when
// they have been copied into testdata (make testdata).
func TestParity(t *testing.T) {
	b, err := bundle.Load("testdata/bundle.json", bundle.Options{})
	if err != nil {
		t.Skipf("no reference bundle in testdata: %v", err)
	}
	e, err := engine.New(b)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/parity_cases.jsonl")
	if err != nil {
		t.Skipf("no parity cases: %v", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	n := 0
	for sc.Scan() {
		var c struct {
			Features map[string]*float64 `json:"features"`
			Expect   engine.Decision     `json:"expect"`
		}
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			t.Fatal(err)
		}
		n++
		in, err := e.Vector(c.Features)
		if err != nil {
			t.Fatal(err)
		}
		got := e.Decide(in)
		for name, w := range c.Expect.Scores {
			g := got.Scores[name]
			if math.Abs(w.Logit-g.Logit) > 1e-9 || w.Alert != g.Alert || len(w.Reasons) != len(g.Reasons) {
				t.Fatalf("case %d %s: want logit %v alert %v, got %v %v", n, name, w.Logit, w.Alert, g.Logit, g.Alert)
			}
			for i := range w.Reasons {
				if w.Reasons[i].Code != g.Reasons[i].Code {
					t.Fatalf("case %d %s reason %d: want %s, got %s", n, name, i, w.Reasons[i].Code, g.Reasons[i].Code)
				}
			}
		}
		if c.Expect.Action.Recommended != got.Action.Recommended || c.Expect.Action.Note != got.Action.Note {
			t.Fatalf("case %d action: want %s (%s), got %s (%s)", n, c.Expect.Action.Recommended, c.Expect.Action.Note, got.Action.Recommended, got.Action.Note)
		}
	}
	if n == 0 {
		t.Fatal("parity file is empty")
	}
	t.Logf("%d reference cases reproduced", n)
}
