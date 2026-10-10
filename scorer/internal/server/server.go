// Package server exposes the engine over HTTP. It only ever listens: there is
// no outbound client anywhere in this binary, so a deployment with egress
// denied loses nothing.
package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/udaykishore-resu/forewarn/scorer/internal/audit"
	"github.com/udaykishore-resu/forewarn/scorer/internal/drift"
	"github.com/udaykishore-resu/forewarn/scorer/internal/engine"
)

type Request struct {
	AccountRef string              `json:"account_ref"`
	AsOf       string              `json:"as_of,omitempty"`
	Features   map[string]*float64 `json:"features"`
}

type Response struct {
	DecisionID   string `json:"decision_id"`
	AccountRef   string `json:"account_ref"`
	ModelVersion string `json:"model_version"`
	BundleSHA256 string `json:"bundle_sha256"`
	engine.Decision
}

type Options struct {
	MaxBodyBytes         int64
	MaxBatch             int
	AuditIncludeFeatures bool
}

type Server struct {
	eng     *engine.Engine
	audit   *audit.Log
	monitor *drift.Monitor
	log     *slog.Logger
	opt     Options
	ready   atomic.Bool
	m       metrics
}

func New(eng *engine.Engine, al *audit.Log, mon *drift.Monitor, log *slog.Logger, opt Options) *Server {
	if opt.MaxBodyBytes == 0 {
		opt.MaxBodyBytes = 4 << 20
	}
	if opt.MaxBatch == 0 {
		opt.MaxBatch = 1000
	}
	s := &Server{eng: eng, audit: al, monitor: mon, log: log, opt: opt}
	s.m.init(eng.Bundle().Policy.Arms)
	s.ready.Store(true)
	return s
}

func (s *Server) SetReady(v bool) { s.ready.Store(v) }

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/score", s.handleScore)
	mux.HandleFunc("POST /v1/score/batch", s.handleBatch)
	mux.HandleFunc("GET /v1/model", s.handleModel)
	mux.HandleFunc("GET /v1/drift", s.handleDrift)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !s.ready.Load() {
			http.Error(w, "not ready", http.StatusServiceUnavailable)
			return
		}
		w.Write([]byte("ready\n"))
	})
	mux.HandleFunc("GET /metrics", s.handleMetrics)
	return mux
}

// InputDigest hashes the feature vector in bundle order, so an audit record can
// be tied to its exact input without storing the input.
func InputDigest(eng *engine.Engine, in engine.Input) string {
	h := sha256.New()
	for i, f := range eng.Bundle().Features {
		if in.Missing[i] {
			fmt.Fprintf(h, "%s=null\n", f.Name)
		} else {
			fmt.Fprintf(h, "%s=%s\n", f.Name, strconv.FormatFloat(in.Value[i], 'g', -1, 64))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// Score runs one request through the engine, the drift monitor and the audit log.
func (s *Server) Score(req Request) (Response, error) {
	if strings.TrimSpace(req.AccountRef) == "" {
		return Response{}, errors.New("account_ref is required")
	}
	in, err := s.eng.Vector(req.Features)
	if err != nil {
		return Response{}, err
	}
	d := s.eng.Decide(in)
	b := s.eng.Bundle()
	logits := map[string]float64{}
	prob := map[string]float64{}
	codes := map[string][]string{}
	for name, sc := range d.Scores {
		logits[name], prob[name] = sc.Logit, sc.Probability
		cs := make([]string, 0, len(sc.Reasons))
		for _, r := range sc.Reasons {
			cs = append(cs, r.Code)
		}
		codes[name] = cs
	}
	if s.monitor != nil {
		s.monitor.Observe(in.Value, in.Missing, logits)
	}
	resp := Response{AccountRef: req.AccountRef, ModelVersion: b.Version, BundleSHA256: b.SHA256, Decision: d}
	if s.audit != nil {
		rec := audit.Record{
			AccountRef: req.AccountRef, AsOf: req.AsOf, BundleSHA256: b.SHA256, InputSHA256: InputDigest(s.eng, in),
			Alert: d.Alert, Driver: d.Driver, Probability: prob, ReasonCodes: codes,
			Action: d.Action.Recommended, ActionNote: d.Action.Note, AdverseActionNotice: d.Action.AdverseActionNotice,
		}
		if s.opt.AuditIncludeFeatures {
			rec.Features = req.Features
		}
		// A decision that cannot be recorded is not returned.
		written, err := s.audit.Append(rec)
		if err != nil {
			return Response{}, fmt.Errorf("audit log: %w", err)
		}
		resp.DecisionID = written.DecisionID
	}
	s.m.observe(d)
	return resp, nil
}

func (s *Server) decode(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, s.opt.MaxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		s.fail(w, http.StatusBadRequest, "invalid request body: "+err.Error())
		return false
	}
	if _, err := dec.Token(); err != io.EOF {
		s.fail(w, http.StatusBadRequest, "request body must contain a single JSON document")
		return false
	}
	return true
}

func (s *Server) fail(w http.ResponseWriter, code int, msg string) {
	s.m.errors.Add(1)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]string{"error": msg})
}

func (s *Server) handleScore(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	var req Request
	if !s.decode(w, r, &req) {
		return
	}
	resp, err := s.Score(req)
	if err != nil {
		code := http.StatusBadRequest
		if strings.HasPrefix(err.Error(), "audit log:") {
			code = http.StatusInternalServerError
			s.log.Error("audit write failed", "error", err.Error())
		}
		s.fail(w, code, err.Error())
		return
	}
	s.m.latency.observe(time.Since(start).Seconds())
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *Server) handleBatch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Items []Request `json:"items"`
	}
	if !s.decode(w, r, &req) {
		return
	}
	if len(req.Items) == 0 || len(req.Items) > s.opt.MaxBatch {
		s.fail(w, http.StatusBadRequest, fmt.Sprintf("items must contain 1 to %d accounts", s.opt.MaxBatch))
		return
	}
	// validate everything before scoring anything, so a batch is all or nothing
	for i, it := range req.Items {
		if strings.TrimSpace(it.AccountRef) == "" {
			s.fail(w, http.StatusBadRequest, fmt.Sprintf("items[%d]: account_ref is required", i))
			return
		}
		if _, err := s.eng.Vector(it.Features); err != nil {
			s.fail(w, http.StatusBadRequest, fmt.Sprintf("items[%d]: %s", i, err))
			return
		}
	}
	out := make([]Response, 0, len(req.Items))
	for _, it := range req.Items {
		resp, err := s.Score(it)
		if err != nil {
			s.log.Error("batch scoring failed", "error", err.Error())
			s.fail(w, http.StatusInternalServerError, err.Error())
			return
		}
		out = append(out, resp)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"items": out})
}

func (s *Server) handleModel(w http.ResponseWriter, _ *http.Request) {
	b := s.eng.Bundle()
	type feat struct {
		Name        string `json:"name"`
		Description string `json:"description"`
		Monotone    int    `json:"monotone"`
		ReasonCode  string `json:"reason_code"`
	}
	fs := make([]feat, 0, len(b.Features))
	for _, f := range b.Features {
		fs = append(fs, feat{f.Name, f.Description, f.Monotone, f.ReasonCode})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"name": b.Name, "version": b.Version, "bundle_sha256": b.SHA256, "created_at": b.CreatedAt,
		"data_source": b.DataSource, "alerts": b.Alerts, "policy_arms": b.Policy.Arms, "features": fs,
	})
}

func (s *Server) handleDrift(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.monitor.Report())
}

// ---------------------------------------------------------------- metrics
// A few counters in Prometheus text format, written by hand so the binary has
// no third-party dependency to review.

type histogram struct {
	mu     sync.Mutex
	bounds []float64
	counts []uint64
	sum    float64
	n      uint64
}

func (h *histogram) observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i, b := range h.bounds {
		if v <= b {
			h.counts[i]++
		}
	}
	h.sum += v
	h.n++
}

type metrics struct {
	decisions atomic.Uint64
	errors    atomic.Uint64
	alerts    map[string]*atomic.Uint64
	actions   map[string]*atomic.Uint64
	adverse   atomic.Uint64
	latency   histogram
}

func (m *metrics) init(arms []string) {
	m.alerts = map[string]*atomic.Uint64{"delinquency": {}, "bankruptcy": {}, "any": {}}
	m.actions = map[string]*atomic.Uint64{}
	for _, a := range arms {
		m.actions[a] = &atomic.Uint64{}
	}
	m.latency.bounds = []float64{0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.05, 0.25}
	m.latency.counts = make([]uint64, len(m.latency.bounds))
}

func (m *metrics) observe(d engine.Decision) {
	m.decisions.Add(1)
	for name, sc := range d.Scores {
		if sc.Alert {
			m.alerts[name].Add(1)
		}
	}
	if d.Alert {
		m.alerts["any"].Add(1)
	}
	if c, ok := m.actions[d.Action.Recommended]; ok {
		c.Add(1)
	}
	if d.Action.AdverseActionNotice {
		m.adverse.Add(1)
	}
}

func (s *Server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	var b strings.Builder
	line := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }
	bd := s.eng.Bundle()
	line("# TYPE forewarn_model_info gauge")
	line(`forewarn_model_info{version=%q,bundle_sha256=%q} 1`, bd.Version, bd.SHA256)
	line("# TYPE forewarn_decisions_total counter")
	line("forewarn_decisions_total %d", s.m.decisions.Load())
	line("# TYPE forewarn_request_errors_total counter")
	line("forewarn_request_errors_total %d", s.m.errors.Load())
	line("# TYPE forewarn_alerts_total counter")
	for _, k := range sortedKeys(s.m.alerts) {
		line(`forewarn_alerts_total{model=%q} %d`, k, s.m.alerts[k].Load())
	}
	line("# TYPE forewarn_actions_total counter")
	for _, k := range sortedKeys(s.m.actions) {
		line(`forewarn_actions_total{action=%q} %d`, k, s.m.actions[k].Load())
	}
	line("# TYPE forewarn_adverse_action_notices_total counter")
	line("forewarn_adverse_action_notices_total %d", s.m.adverse.Load())
	h := &s.m.latency
	h.mu.Lock()
	line("# TYPE forewarn_score_duration_seconds histogram")
	for i, bound := range h.bounds {
		line(`forewarn_score_duration_seconds_bucket{le="%g"} %d`, bound, h.counts[i])
	}
	line(`forewarn_score_duration_seconds_bucket{le="+Inf"} %d`, h.n)
	line("forewarn_score_duration_seconds_sum %g", h.sum)
	line("forewarn_score_duration_seconds_count %d", h.n)
	h.mu.Unlock()
	if s.monitor != nil {
		rep := s.monitor.Report()
		line("# HELP forewarn_psi Population stability index of scored accounts against the training reference.")
		line("# TYPE forewarn_psi gauge")
		for _, it := range rep.Features {
			line(`forewarn_psi{kind="feature",name=%q} %g`, it.Name, it.PSI)
		}
		for _, it := range rep.Scores {
			line(`forewarn_psi{kind="score",name=%q} %g`, it.Name, it.PSI)
		}
		line("# TYPE forewarn_drift_observations gauge")
		line("forewarn_drift_observations %d", rep.N)
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	io.WriteString(w, b.String())
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
