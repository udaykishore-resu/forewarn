package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/udaykishore-resu/forewarn/scorer/internal/audit"
	"github.com/udaykishore-resu/forewarn/scorer/internal/drift"
	"github.com/udaykishore-resu/forewarn/scorer/internal/engine"
	"github.com/udaykishore-resu/forewarn/scorer/internal/testutil"
)

func newServer(t *testing.T) (*httptest.Server, string) {
	t.Helper()
	b, err := testutil.Bundle(t)
	if err != nil {
		t.Fatal(err)
	}
	e, err := engine.New(b)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	al, err := audit.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { al.Close() })
	s := New(e, al, drift.NewMonitor(b), slog.New(slog.NewJSONHandler(io.Discard, nil)), Options{MaxBatch: 3})
	ts := httptest.NewServer(s.Handler())
	t.Cleanup(ts.Close)
	return ts, path
}

func post(t *testing.T, url, body string) (int, map[string]any) {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp.StatusCode, out
}

const good = `{"account_ref":"a-1","features":{"utilization":0.9,"util_change":0.2,"buffer_days":5,"open_to_buy":1500}}`

func TestScoreEndpoint(t *testing.T) {
	ts, auditPath := newServer(t)
	code, out := post(t, ts.URL+"/v1/score", good)
	if code != 200 {
		t.Fatalf("status %d: %v", code, out)
	}
	if out["alert"] != true || out["driver"] != "delinquency" || out["decision_id"] == "" || len(out["bundle_sha256"].(string)) != 64 {
		t.Fatalf("unexpected response: %v", out)
	}
	action := out["action"].(map[string]any)
	if action["recommended"] != "limit_decrease" || action["adverse_action_notice"] != true {
		t.Fatalf("unexpected action: %v", action)
	}
	// the decision is in the audit log, and the log holds no feature values
	if n, _, err := audit.Verify(auditPath); err != nil || n != 1 {
		t.Fatalf("audit log: %d records, %v", n, err)
	}
}

func TestBadRequestsAreRejected(t *testing.T) {
	ts, auditPath := newServer(t)
	for name, body := range map[string]string{
		"unknown feature":     `{"account_ref":"a","features":{"zip_code":20170}}`,
		"unknown field":       `{"account_ref":"a","features":{},"override_alert":false}`,
		"missing account_ref": `{"features":{"utilization":0.5}}`,
		"string for a number": `{"account_ref":"a","features":{"utilization":"high"}}`,
		"two documents":       good + good,
		"not json":            `utilization=0.9`,
	} {
		if code, out := post(t, ts.URL+"/v1/score", body); code != 400 || out["error"] == nil {
			t.Errorf("%s: status %d, body %v; want 400 with an error", name, code, out)
		}
	}
	if n, _, _ := audit.Verify(auditPath); n != 0 {
		t.Fatalf("rejected requests wrote %d audit records", n)
	}
	resp, _ := http.Get(ts.URL + "/v1/score")
	if resp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/score = %d, want 405", resp.StatusCode)
	}
}

func TestBatchIsAllOrNothing(t *testing.T) {
	ts, auditPath := newServer(t)
	bad := `{"account_ref":"a-2","features":{"zip_code":1}}`
	if code, _ := post(t, ts.URL+"/v1/score/batch", `{"items":[`+good+`,`+bad+`]}`); code != 400 {
		t.Fatalf("batch with one bad item: status %d, want 400", code)
	}
	if n, _, _ := audit.Verify(auditPath); n != 0 {
		t.Fatalf("a rejected batch scored %d accounts", n)
	}
	if code, _ := post(t, ts.URL+"/v1/score/batch", `{"items":[`+good+`,`+good+`,`+good+`,`+good+`]}`); code != 400 {
		t.Fatalf("batch over the limit: status %d, want 400", code)
	}
	code, out := post(t, ts.URL+"/v1/score/batch", `{"items":[`+good+`,`+good+`]}`)
	if code != 200 || len(out["items"].([]any)) != 2 {
		t.Fatalf("valid batch: status %d, %v", code, out)
	}
}

func TestMetricsAndModelEndpoints(t *testing.T) {
	ts, _ := newServer(t)
	post(t, ts.URL+"/v1/score", good)
	resp, err := http.Get(ts.URL + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	for _, want := range []string{
		"forewarn_decisions_total 1",
		`forewarn_alerts_total{model="delinquency"} 1`,
		`forewarn_actions_total{action="limit_decrease"} 1`,
		"forewarn_adverse_action_notices_total 1",
		`forewarn_psi{kind="feature",name="utilization"}`,
	} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("metrics missing %q", want)
		}
	}
	resp, _ = http.Get(ts.URL + "/v1/model")
	var model map[string]any
	json.NewDecoder(resp.Body).Decode(&model)
	resp.Body.Close()
	if model["version"] != "test" || len(model["features"].([]any)) != 4 {
		t.Fatalf("unexpected model description: %v", model)
	}
	for _, path := range []string{"/healthz", "/readyz"} {
		if resp, _ := http.Get(ts.URL + path); resp.StatusCode != 200 {
			t.Errorf("%s = %d", path, resp.StatusCode)
		}
	}
}
