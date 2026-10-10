// Package bundle loads and validates the model bundle produced by the training
// pipeline. Validation is deliberately strict: the scorer refuses to start on a
// bundle that breaks a promise made in the model documentation (a prohibited
// feature, a table that is not monotone in its declared direction, a malformed
// threshold list), so those promises are enforced at runtime and not only on
// paper.
package bundle

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

const SchemaVersion = 1

type Feature struct {
	Name        string `json:"name"`
	Group       string `json:"group"`
	Source      string `json:"source"`
	Monotone    int    `json:"monotone"`
	ReasonCode  string `json:"reason_code"`
	ReasonText  string `json:"reason_text"`
	MissingText string `json:"missing_text"`
	Description string `json:"description"`
	Status      string `json:"status"`
	Note        string `json:"note"`
	// Required marks a feature that was never missing in training. The model has
	// no learned behaviour for its absence, so the scorer rejects a request
	// without it instead of inventing one.
	Required bool `json:"required"`
}

type Term struct {
	Feature    string    `json:"feature"`
	Thresholds []float64 `json:"thresholds"`
	Values     []float64 `json:"values"`
	Missing    float64   `json:"missing"`
}

type Calibration struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

type Model struct {
	Name        string         `json:"name"`
	Link        string         `json:"link"`
	Intercept   float64        `json:"intercept"`
	Calibration Calibration    `json:"calibration"`
	Terms       []Term         `json:"terms"`
	Meta        map[string]any `json:"meta"`
}

type Models struct {
	Delinquency  Model            `json:"delinquency"`
	Bankruptcy   Model            `json:"bankruptcy"`
	ValueEffects map[string]Model `json:"value_effects"`
}

type Alert struct {
	ThresholdProbability float64 `json:"threshold_probability"`
	ThresholdLogit       float64 `json:"threshold_logit"`
	ShareOnValidation    float64 `json:"share_alerted_on_validation"`
}

type Condition struct {
	Feature string  `json:"feature"`
	Op      string  `json:"op"`
	Value   float64 `json:"value"`
}

type Policy struct {
	Arms           []string               `json:"arms"`
	Eligibility    map[string][]Condition `json:"eligibility"`
	PilotSupport   [][]Condition          `json:"pilot_support"`
	MinGain        float64                `json:"min_gain"`
	AdverseActions []string               `json:"adverse_actions"`
	HorizonMonths  int                    `json:"value_horizon_months"`
}

type Reasons struct {
	MaxReasons int     `json:"max_reasons"`
	MinPoints  float64 `json:"min_points"`
}

type Reference struct {
	Edges  []float64 `json:"edges"`
	Shares []float64 `json:"shares"`
}

type Drift struct {
	Features map[string]Reference `json:"features"`
	Scores   map[string]Reference `json:"scores"`
}

type Bundle struct {
	SchemaVersion  int              `json:"schema_version"`
	Name           string           `json:"name"`
	Version        string           `json:"version"`
	CreatedAt      string           `json:"created_at"`
	DataSource     string           `json:"data_source"`
	Training       json.RawMessage  `json:"training"`
	Features       []Feature        `json:"features"`
	Models         Models           `json:"models"`
	Alerts         map[string]Alert `json:"alerts"`
	Reasons        Reasons          `json:"reasons"`
	Policy         Policy           `json:"policy"`
	DriftReference Drift            `json:"drift_reference"`

	// SHA256 is the digest of the file as read from disk. It identifies the
	// model in every decision and audit record.
	SHA256 string `json:"-"`
}

// Options controls validation.
type Options struct {
	// ExpectSHA256, when set, must equal the digest of the file. Deployments pin
	// it so that a swapped or edited bundle fails closed.
	ExpectSHA256 string
	// AllowRestricted permits features with status "restricted". Prohibited
	// features are never permitted.
	AllowRestricted bool
}

func Load(path string, opt Options) (*Bundle, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read bundle: %w", err)
	}
	return Parse(raw, opt)
}

// Parse validates a bundle held in memory.
func Parse(raw []byte, opt Options) (*Bundle, error) {
	sum := sha256.Sum256(raw)
	digest := hex.EncodeToString(sum[:])
	if opt.ExpectSHA256 != "" && opt.ExpectSHA256 != digest {
		return nil, fmt.Errorf("bundle digest mismatch: file is %s, expected %s", digest, opt.ExpectSHA256)
	}
	var b Bundle
	if err := json.Unmarshal(raw, &b); err != nil {
		return nil, fmt.Errorf("parse bundle: %w", err)
	}
	b.SHA256 = digest
	if err := b.Validate(opt); err != nil {
		return nil, fmt.Errorf("invalid bundle: %w", err)
	}
	return &b, nil
}

func (b *Bundle) Validate(opt Options) error {
	if b.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schema_version %d not supported (want %d)", b.SchemaVersion, SchemaVersion)
	}
	if len(b.Features) == 0 {
		return fmt.Errorf("no features")
	}
	feats := map[string]Feature{}
	for _, f := range b.Features {
		if _, dup := feats[f.Name]; dup {
			return fmt.Errorf("duplicate feature %q", f.Name)
		}
		switch f.Status {
		case "allowed":
		case "restricted":
			if !opt.AllowRestricted {
				return fmt.Errorf("feature %q is restricted; start with -allow-restricted only with documented compliance sign-off", f.Name)
			}
		default:
			return fmt.Errorf("feature %q has status %q and may not be used", f.Name, f.Status)
		}
		if f.Monotone < -1 || f.Monotone > 1 {
			return fmt.Errorf("feature %q: monotone must be -1, 0 or 1", f.Name)
		}
		feats[f.Name] = f
	}
	risk := map[string]*Model{"delinquency": &b.Models.Delinquency, "bankruptcy": &b.Models.Bankruptcy}
	for name, m := range risk {
		if m.Link != "logit" {
			return fmt.Errorf("model %s: link %q, want logit", name, m.Link)
		}
		// a non-positive slope would reverse every declared direction after calibration
		if m.Calibration.B <= 0 {
			return fmt.Errorf("model %s: calibration slope must be positive, got %v", name, m.Calibration.B)
		}
		if err := validateModel(name, m, feats, true); err != nil {
			return err
		}
		a, ok := b.Alerts[name]
		if !ok {
			return fmt.Errorf("no alert threshold for %s", name)
		}
		if !finite(a.ThresholdLogit) || a.ThresholdProbability <= 0 || a.ThresholdProbability >= 1 {
			return fmt.Errorf("alert threshold for %s is out of range", name)
		}
		if math.Abs(math.Log(a.ThresholdProbability/(1-a.ThresholdProbability))-a.ThresholdLogit) > 1e-9 {
			return fmt.Errorf("alert threshold for %s: probability and logit disagree", name)
		}
	}
	p := b.Policy
	if len(p.Arms) == 0 || p.Arms[0] != "none" {
		return fmt.Errorf("policy.arms must start with \"none\"")
	}
	for _, arm := range p.Arms {
		m, ok := b.Models.ValueEffects[arm]
		if !ok {
			return fmt.Errorf("no value-effect model for arm %q", arm)
		}
		if m.Link != "identity" {
			return fmt.Errorf("value-effect model %s: link %q, want identity", arm, m.Link)
		}
		if err := validateModel("value_effects."+arm, &m, feats, false); err != nil {
			return err
		}
	}
	checkCond := func(where string, c Condition) error {
		if _, ok := feats[c.Feature]; !ok {
			return fmt.Errorf("%s: unknown feature %q", where, c.Feature)
		}
		switch c.Op {
		case "<=", ">=", "<", ">":
		default:
			return fmt.Errorf("%s: unsupported operator %q", where, c.Op)
		}
		if !finite(c.Value) {
			return fmt.Errorf("%s: non-finite value", where)
		}
		return nil
	}
	for arm, conds := range p.Eligibility {
		if _, ok := b.Models.ValueEffects[arm]; !ok {
			return fmt.Errorf("eligibility for unknown arm %q", arm)
		}
		for _, c := range conds {
			if err := checkCond("eligibility."+arm, c); err != nil {
				return err
			}
		}
	}
	for _, clause := range p.PilotSupport {
		for _, c := range clause {
			if err := checkCond("pilot_support", c); err != nil {
				return err
			}
		}
	}
	if !finite(p.MinGain) || p.MinGain < 0 {
		return fmt.Errorf("policy.min_gain must be a non-negative number")
	}
	if b.Reasons.MaxReasons < 1 || !finite(b.Reasons.MinPoints) {
		return fmt.Errorf("reasons configuration is invalid")
	}
	for name, r := range b.DriftReference.Features {
		if _, ok := feats[name]; !ok {
			return fmt.Errorf("drift reference for unknown feature %q", name)
		}
		if err := validateReference(name, r); err != nil {
			return err
		}
	}
	for name, r := range b.DriftReference.Scores {
		if err := validateReference("score "+name, r); err != nil {
			return err
		}
	}
	return nil
}

func validateModel(name string, m *Model, feats map[string]Feature, enforceMonotone bool) error {
	if !finite(m.Intercept) || !finite(m.Calibration.A) || !finite(m.Calibration.B) {
		return fmt.Errorf("model %s: non-finite intercept or calibration", name)
	}
	seen := map[string]bool{}
	for _, t := range m.Terms {
		f, ok := feats[t.Feature]
		if !ok {
			return fmt.Errorf("model %s: term on unknown feature %q", name, t.Feature)
		}
		if seen[t.Feature] {
			return fmt.Errorf("model %s: feature %q appears twice", name, t.Feature)
		}
		seen[t.Feature] = true
		if len(t.Values) != len(t.Thresholds)+1 {
			return fmt.Errorf("model %s.%s: %d values for %d thresholds", name, t.Feature, len(t.Values), len(t.Thresholds))
		}
		if !finite(t.Missing) {
			return fmt.Errorf("model %s.%s: non-finite missing value", name, t.Feature)
		}
		for i, v := range t.Thresholds {
			if !finite(v) || (i > 0 && v <= t.Thresholds[i-1]) {
				return fmt.Errorf("model %s.%s: thresholds must be finite and strictly increasing", name, t.Feature)
			}
		}
		for i, v := range t.Values {
			if !finite(v) {
				return fmt.Errorf("model %s.%s: non-finite value", name, t.Feature)
			}
			if !enforceMonotone || i == 0 {
				continue
			}
			d := v - t.Values[i-1]
			if (f.Monotone > 0 && d < -1e-12) || (f.Monotone < 0 && d > 1e-12) {
				return fmt.Errorf("model %s.%s: table is not monotone in its declared direction (%+d)", name, t.Feature, f.Monotone)
			}
		}
	}
	return nil
}

func validateReference(name string, r Reference) error {
	if len(r.Shares) != len(r.Edges)+2 {
		return fmt.Errorf("drift reference %s: want %d shares, got %d", name, len(r.Edges)+2, len(r.Shares))
	}
	for i, v := range r.Edges {
		if !finite(v) || (i > 0 && v <= r.Edges[i-1]) {
			return fmt.Errorf("drift reference %s: edges must be finite and strictly increasing", name)
		}
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
