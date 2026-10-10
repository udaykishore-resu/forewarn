package drift

import (
	"math"
	"testing"

	"github.com/udaykishore-resu/forewarn/scorer/internal/testutil"
)

func TestPSI(t *testing.T) {
	b, err := testutil.Bundle(t)
	if err != nil {
		t.Fatal(err)
	}
	// reference shares for utilization: 50% <= 0.3, 30% in (0.3, 0.8], 20% above, none missing
	same := NewMonitor(b)
	for i := 0; i < 1000; i++ {
		x := 0.9
		if i < 500 {
			x = 0.1
		} else if i < 800 {
			x = 0.5
		}
		same.Observe([]float64{x, 0, 0, 0}, []bool{false, false, false, false}, map[string]float64{"delinquency": -3})
	}
	rep := same.Report()
	if rep.Features[0].Name != "utilization" || rep.Features[0].PSI > 1e-6 || rep.Features[0].Status != "ok" {
		t.Fatalf("identical population should have PSI 0: %+v", rep.Features[0])
	}

	// everything in the top bin: PSI = sum (c - r) ln(c / r) with the 1e-4 floor
	shifted := NewMonitor(b)
	for i := 0; i < 1000; i++ {
		shifted.Observe([]float64{0.95, 0, 0, 0}, []bool{false, false, false, false}, nil)
	}
	got := shifted.Report().Features[0]
	r := []float64{0.5, 0.3, 0.2, 1e-4}
	c := []float64{1e-4, 1e-4, 1.0, 1e-4}
	rs, cs := 0.0, 0.0
	for i := range r {
		rs, cs = rs+r[i], cs+c[i]
	}
	want := 0.0
	for i := range r {
		want += (c[i]/cs - r[i]/rs) * math.Log((c[i]/cs)/(r[i]/rs))
	}
	if math.Abs(got.PSI-want) > 1e-12 || got.Status != "alert" {
		t.Fatalf("PSI = %v (%s), want %v (alert)", got.PSI, got.Status, want)
	}
	// an input moved, no score was observed: the two statuses are reported separately
	if rep := shifted.Report(); rep.FeatureStatus != "alert" || rep.ScoreStatus != "ok" || rep.Worst != "alert" {
		t.Fatalf("statuses = feature %s, score %s, worst %s", rep.FeatureStatus, rep.ScoreStatus, rep.Worst)
	}

	// a value on a bin edge counts in the lower bin; missing values have their own bin
	edge := NewMonitor(b)
	edge.Observe([]float64{0.3, 0, 0, 0}, []bool{false, false, false, false}, nil)
	edge.Observe([]float64{0, 0, 0, 0}, []bool{true, false, false, false}, nil)
	s := edge.features[0]
	if s.counts[0] != 1 || s.counts[3] != 1 {
		t.Fatalf("bin counts = %v, want one in the first bin and one missing", s.counts)
	}
}
