// Package drift measures how far the scored population has moved from the
// training population, using the population stability index (PSI) per feature
// and per score. Reference bins travel inside the model bundle, so monitoring
// needs no access to training data.
package drift

import (
	"math"
	"sort"
	"sync"

	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
)

const (
	Watch = 0.10 // conventional PSI thresholds
	Alert = 0.25
	eps   = 1e-4
)

type series struct {
	ref    bundle.Reference
	counts []float64 // len(edges)+2; last bin is "missing"
	n      float64
}

func (s *series) add(x float64, missing bool) {
	if missing {
		s.counts[len(s.counts)-1]++
	} else {
		// bin i covers (edges[i-1], edges[i]], the same rule the scorer uses
		s.counts[sort.SearchFloat64s(s.ref.Edges, x)]++
	}
	s.n++
}

func (s *series) psi() float64 {
	if s.n == 0 {
		return math.NaN()
	}
	rs, cs := 0.0, 0.0
	r := make([]float64, len(s.counts))
	c := make([]float64, len(s.counts))
	for i := range s.counts {
		r[i] = math.Max(s.ref.Shares[i], eps)
		c[i] = math.Max(s.counts[i]/s.n, eps)
		rs += r[i]
		cs += c[i]
	}
	v := 0.0
	for i := range r {
		ri, ci := r[i]/rs, c[i]/cs
		v += (ci - ri) * math.Log(ci/ri)
	}
	return v
}

// Monitor accumulates counts; it is safe for concurrent use.
type Monitor struct {
	mu       sync.Mutex
	names    []string
	features []*series
	scores   map[string]*series
}

func NewMonitor(b *bundle.Bundle) *Monitor {
	m := &Monitor{scores: map[string]*series{}}
	for _, f := range b.Features {
		ref, ok := b.DriftReference.Features[f.Name]
		m.names = append(m.names, f.Name)
		if !ok {
			m.features = append(m.features, nil)
			continue
		}
		m.features = append(m.features, &series{ref: ref, counts: make([]float64, len(ref.Shares))})
	}
	for name, ref := range b.DriftReference.Scores {
		m.scores[name] = &series{ref: ref, counts: make([]float64, len(ref.Shares))}
	}
	return m
}

// Observe records one scored account: its feature vector and score logits.
func (m *Monitor) Observe(values []float64, missing []bool, logits map[string]float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i, s := range m.features {
		if s != nil {
			s.add(values[i], missing[i])
		}
	}
	for name, z := range logits {
		if s, ok := m.scores[name]; ok {
			s.add(z, false)
		}
	}
}

type Item struct {
	Name         string  `json:"name"`
	PSI          float64 `json:"psi"`
	Status       string  `json:"status"`
	MissingShare float64 `json:"missing_share"`
	RefMissing   float64 `json:"reference_missing_share"`
}

// Report separates scores from inputs. Score drift means the model's output
// distribution has moved and is the signal to act on. Input drift is the
// diagnostic that explains it; single inputs also move for benign reasons
// (December spending, a 31-day month) without moving the scores.
type Report struct {
	N             int    `json:"n"`
	Features      []Item `json:"features"`
	Scores        []Item `json:"scores"`
	ScoreStatus   string `json:"score_status"`
	FeatureStatus string `json:"feature_status"`
	Worst         string `json:"worst_status"`
}

func status(psi float64) string {
	switch {
	case math.IsNaN(psi):
		return "no_data"
	case psi >= Alert:
		return "alert"
	case psi >= Watch:
		return "watch"
	default:
		return "ok"
	}
}

func (m *Monitor) Report() Report {
	m.mu.Lock()
	defer m.mu.Unlock()
	rep := Report{Worst: "ok", ScoreStatus: "ok", FeatureStatus: "ok"}
	rank := map[string]int{"no_data": 0, "ok": 0, "watch": 1, "alert": 2}
	item := func(name string, s *series, group *string) Item {
		p := s.psi()
		it := Item{Name: name, PSI: p, Status: status(p), RefMissing: s.ref.Shares[len(s.ref.Shares)-1]}
		if s.n > 0 {
			it.MissingShare = s.counts[len(s.counts)-1] / s.n
		}
		if math.IsNaN(p) {
			it.PSI = 0
		}
		if rank[it.Status] > rank[rep.Worst] {
			rep.Worst = it.Status
		}
		if rank[it.Status] > rank[*group] {
			*group = it.Status
		}
		if int(s.n) > rep.N {
			rep.N = int(s.n)
		}
		return it
	}
	for i, s := range m.features {
		if s != nil {
			rep.Features = append(rep.Features, item(m.names[i], s, &rep.FeatureStatus))
		}
	}
	var names []string
	for n := range m.scores {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		rep.Scores = append(rep.Scores, item(n, m.scores[n], &rep.ScoreStatus))
	}
	return rep
}
