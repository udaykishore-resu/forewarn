// Package engine turns one account's features into scores, alerts, reasons and
// a recommended action. It is the second implementation of
// ml/forewarn/score.py; `forewarn verify` replays reference cases through it.
//
// The risk models are additive: each feature has a lookup table, a score is
// the intercept plus one table value per feature. Scoring is therefore a
// binary search per feature, and the contribution of each feature to a
// decision is exact.
package engine

import (
	"fmt"
	"math"
	"sort"

	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
)

// Input is a feature vector in bundle order.
type Input struct {
	Value   []float64
	Missing []bool
}

type Reason struct {
	Code    string  `json:"code"`
	Text    string  `json:"text"`
	Feature string  `json:"feature"`
	Points  float64 `json:"points"`
}

type Score struct {
	Probability float64  `json:"probability"`
	Logit       float64  `json:"logit"`
	Alert       bool     `json:"alert"`
	Reasons     []Reason `json:"reasons"`
}

type Action struct {
	Recommended         string             `json:"recommended"`
	Note                string             `json:"note"`
	EstimatedGain       map[string]float64 `json:"estimated_gain"`
	Eligible            map[string]bool    `json:"eligible"`
	InPilotSupport      bool               `json:"in_pilot_support"`
	AdverseActionNotice bool               `json:"adverse_action_notice"`
}

type Decision struct {
	Alert  bool             `json:"alert"`
	Driver string           `json:"driver"`
	Scores map[string]Score `json:"scores"`
	Action Action           `json:"action"`
}

type term struct {
	idx        int
	feature    string
	thresholds []float64
	values     []float64
	missing    float64
}

type model struct {
	intercept float64
	a, b      float64
	terms     []term
}

type cond struct {
	idx int
	op  string
	v   float64
}

type Engine struct {
	b           *bundle.Bundle
	index       map[string]int
	risk        map[string]*model
	riskOrder   []string
	effects     []*model
	eligibility [][]cond
	support     [][]cond
	adverse     map[string]bool
	groupOfTerm map[string]string
}

func New(b *bundle.Bundle) (*Engine, error) {
	e := &Engine{
		b:           b,
		index:       map[string]int{},
		risk:        map[string]*model{},
		riskOrder:   []string{"delinquency", "bankruptcy"},
		adverse:     map[string]bool{},
		groupOfTerm: map[string]string{},
	}
	for i, f := range b.Features {
		e.index[f.Name] = i
		e.groupOfTerm[f.Name] = f.Group
	}
	compile := func(m *bundle.Model) (*model, error) {
		out := &model{intercept: m.Intercept, a: m.Calibration.A, b: m.Calibration.B}
		for _, t := range m.Terms {
			i, ok := e.index[t.Feature]
			if !ok {
				return nil, fmt.Errorf("term on unknown feature %q", t.Feature)
			}
			out.terms = append(out.terms, term{i, t.Feature, t.Thresholds, t.Values, t.Missing})
		}
		return out, nil
	}
	var err error
	if e.risk["delinquency"], err = compile(&b.Models.Delinquency); err != nil {
		return nil, err
	}
	if e.risk["bankruptcy"], err = compile(&b.Models.Bankruptcy); err != nil {
		return nil, err
	}
	conds := func(in []bundle.Condition) []cond {
		out := make([]cond, len(in))
		for i, c := range in {
			out[i] = cond{e.index[c.Feature], c.Op, c.Value}
		}
		return out
	}
	for _, arm := range b.Policy.Arms {
		m := b.Models.ValueEffects[arm]
		cm, err := compile(&m)
		if err != nil {
			return nil, err
		}
		e.effects = append(e.effects, cm)
		e.eligibility = append(e.eligibility, conds(b.Policy.Eligibility[arm]))
	}
	for _, clause := range b.Policy.PilotSupport {
		e.support = append(e.support, conds(clause))
	}
	for _, a := range b.Policy.AdverseActions {
		e.adverse[a] = true
	}
	return e, nil
}

func (e *Engine) Bundle() *bundle.Bundle { return e.b }

// Vector converts a name->value map into bundle order. A nil value or an absent
// name is a missing value, which is accepted only for features that were
// sometimes missing in training. Unknown names, non-finite numbers and absent
// required features are errors: the scorer never guesses what a caller meant.
func (e *Engine) Vector(features map[string]*float64) (Input, error) {
	in := Input{Value: make([]float64, len(e.b.Features)), Missing: make([]bool, len(e.b.Features))}
	for i := range in.Missing {
		in.Missing[i] = true
	}
	for name, v := range features {
		i, ok := e.index[name]
		if !ok {
			return in, fmt.Errorf("unknown feature %q", name)
		}
		if v == nil {
			continue
		}
		if math.IsNaN(*v) || math.IsInf(*v, 0) {
			return in, fmt.Errorf("feature %q is not a finite number", name)
		}
		in.Value[i], in.Missing[i] = *v, false
	}
	for i, f := range e.b.Features {
		if f.Required && in.Missing[i] {
			return in, fmt.Errorf("feature %q is required", f.Name)
		}
	}
	return in, nil
}

func (t *term) contrib(in Input) float64 {
	if in.Missing[t.idx] {
		return t.missing
	}
	// bin i covers (thresholds[i-1], thresholds[i]]
	return t.values[sort.SearchFloat64s(t.thresholds, in.Value[t.idx])]
}

// raw sums contributions in term order. The order is part of the contract
// with the reference implementation.
func (m *model) raw(in Input, contrib []float64) float64 {
	r := m.intercept
	for j := range m.terms {
		c := m.terms[j].contrib(in)
		if contrib != nil {
			contrib[j] = c
		}
		r += c
	}
	return r
}

func holds(c cond, in Input) bool {
	if in.Missing[c.idx] {
		return false
	}
	x := in.Value[c.idx]
	switch c.op {
	case "<=":
		return x <= c.v
	case ">=":
		return x >= c.v
	case "<":
		return x < c.v
	default:
		return x > c.v
	}
}

// reasons returns the principal reasons for a score, one per reason group.
// Terms are centred on the training population, so a contribution is the
// number of log-odds points by which this account is riskier than the average
// account on that feature. The groups adding the most risk are the reasons.
func (e *Engine) reasons(m *model, contrib []float64, in Input) []Reason {
	type group struct {
		points, top float64
		order       int
		term        int
	}
	groups := map[string]*group{}
	var names []string
	for j := range m.terms {
		t := &m.terms[j]
		lost := contrib[j]
		gname := e.groupOfTerm[t.feature]
		g, ok := groups[gname]
		if !ok {
			g = &group{top: math.Inf(-1), order: j, term: -1}
			groups[gname] = g
			names = append(names, gname)
		}
		g.points += lost
		if lost > g.top {
			g.top, g.term = lost, j
		}
	}
	var ranked []*group
	for _, n := range names {
		if groups[n].points > e.b.Reasons.MinPoints {
			ranked = append(ranked, groups[n])
		}
	}
	sort.SliceStable(ranked, func(i, k int) bool {
		if ranked[i].points != ranked[k].points {
			return ranked[i].points > ranked[k].points
		}
		return ranked[i].order < ranked[k].order
	})
	if len(ranked) > e.b.Reasons.MaxReasons {
		ranked = ranked[:e.b.Reasons.MaxReasons]
	}
	out := make([]Reason, 0, len(ranked))
	for _, g := range ranked {
		t := &m.terms[g.term]
		f := e.b.Features[t.idx]
		r := Reason{Code: f.ReasonCode, Text: f.ReasonText, Feature: f.Name, Points: g.points}
		if in.Missing[t.idx] {
			r.Code, r.Text = f.ReasonCode+"M", f.MissingText
		}
		out = append(out, r)
	}
	return out
}

// Decide scores one account.
func (e *Engine) Decide(in Input) Decision {
	d := Decision{Scores: map[string]Score{}}
	margin := map[string]float64{}
	for _, name := range e.riskOrder {
		m := e.risk[name]
		contrib := make([]float64, len(m.terms))
		raw := m.raw(in, contrib)
		// The explicit conversion stops the compiler fusing the multiply and
		// add on arm64, which would change the last bit of the result.
		z := m.a + float64(m.b*raw)
		thr := e.b.Alerts[name].ThresholdLogit
		alert := z >= thr
		d.Scores[name] = Score{
			Probability: 1.0 / (1.0 + math.Exp(-z)),
			Logit:       z,
			Alert:       alert,
			Reasons:     e.reasons(m, contrib, in),
		}
		margin[name] = z - thr
		d.Alert = d.Alert || alert
	}
	if d.Alert {
		d.Driver = "bankruptcy"
		if margin["delinquency"] >= margin["bankruptcy"] {
			d.Driver = "delinquency"
		}
	}

	arms := e.b.Policy.Arms
	act := Action{EstimatedGain: map[string]float64{}, Eligible: map[string]bool{}}
	g0 := e.effects[0].raw(in, nil)
	best, bestGain := 0, math.Inf(-1)
	for j, arm := range arms {
		gain := e.effects[j].raw(in, nil) - g0
		ok := true
		for _, c := range e.eligibility[j] {
			ok = ok && holds(c, in)
		}
		act.EstimatedGain[arm], act.Eligible[arm] = gain, ok
		if j > 0 && ok && gain > bestGain {
			best, bestGain = j, gain
		}
	}
	for _, clause := range e.support {
		all := true
		for _, c := range clause {
			all = all && holds(c, in)
		}
		if all {
			act.InPilotSupport = true
			break
		}
	}
	clears := bestGain >= e.b.Policy.MinGain
	switch {
	case !d.Alert:
		act.Recommended, act.Note = arms[0], "below_alert_threshold"
	case !act.InPilotSupport:
		act.Recommended, act.Note = arms[0], "outside_pilot_support"
	case !clears:
		act.Recommended, act.Note = arms[0], "no_action_clears_minimum_gain"
	default:
		act.Recommended, act.Note = arms[best], "recommended"
	}
	act.AdverseActionNotice = e.adverse[act.Recommended]
	d.Action = act
	return d
}
