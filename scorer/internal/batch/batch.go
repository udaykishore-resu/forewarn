// Package batch scores a JSON-lines file of accounts, which is how an early
// warning run normally happens: once a night or once a week over the whole
// book, inside the bank's own environment.
package batch

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"

	"github.com/udaykishore-resu/forewarn/scorer/internal/drift"
	"github.com/udaykishore-resu/forewarn/scorer/internal/server"
)

type Summary struct {
	Accounts      int            `json:"accounts"`
	Alerts        map[string]int `json:"alerts"`
	Actions       map[string]int `json:"actions"`
	AdverseNotice int            `json:"adverse_action_notices"`
	BundleSHA256  string         `json:"bundle_sha256"`
	// AuditRecords and AuditHead identify the end of the audit log after this
	// run. Stored away from the log, they make later truncation detectable.
	AuditRecords uint64       `json:"audit_records,omitempty"`
	AuditHead    string       `json:"audit_head,omitempty"`
	Drift        drift.Report `json:"drift"`
}

// Run reads one request per line and writes one decision per line. It stops at
// the first malformed line: a partially scored book is worse than a failed run.
func Run(srv *server.Server, mon *drift.Monitor, bundleSHA string, in io.Reader, out io.Writer) (Summary, error) {
	sum := Summary{Alerts: map[string]int{}, Actions: map[string]int{}, BundleSHA256: bundleSHA}
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	w := bufio.NewWriter(out)
	defer w.Flush()
	enc := json.NewEncoder(w)
	line := 0
	for sc.Scan() {
		line++
		if len(sc.Bytes()) == 0 {
			continue
		}
		var req server.Request
		if err := json.Unmarshal(sc.Bytes(), &req); err != nil {
			return sum, fmt.Errorf("line %d: %w", line, err)
		}
		resp, err := srv.Score(req)
		if err != nil {
			return sum, fmt.Errorf("line %d: %w", line, err)
		}
		if err := enc.Encode(resp); err != nil {
			return sum, err
		}
		sum.Accounts++
		for name, s := range resp.Scores {
			if s.Alert {
				sum.Alerts[name]++
			}
		}
		if resp.Alert {
			sum.Alerts["any"]++
		}
		sum.Actions[resp.Action.Recommended]++
		if resp.Action.AdverseActionNotice {
			sum.AdverseNotice++
		}
	}
	if err := sc.Err(); err != nil {
		return sum, err
	}
	sum.Drift = mon.Report()
	return sum, nil
}
