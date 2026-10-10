// Package audit writes one record per decision to an append-only JSON-lines
// file. Each record carries the hash of the previous one, so editing, removing
// or reordering a past decision breaks every hash after it and `forewarn
// audit-verify` reports where.
//
// What the chain cannot do by itself: it is unkeyed, so someone able to rewrite
// the whole file can recompute it, and dropping records from the end leaves a
// shorter chain that is still valid. Both are closed by keeping the head (the
// last sequence number and hash) somewhere the writer of the file cannot
// change: the batch summary records it, and `audit-verify -expect-head` checks
// a log against it. Write-once storage for the file itself is the other half.
//
// Records hold the model digest, a digest of the input, the outputs and the
// reason codes. Feature values are not written unless explicitly enabled.
package audit

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

const genesis = "0000000000000000000000000000000000000000000000000000000000000000"

type Record struct {
	Seq                 uint64              `json:"seq"`
	Time                string              `json:"time"`
	DecisionID          string              `json:"decision_id"`
	AccountRef          string              `json:"account_ref"`
	AsOf                string              `json:"as_of,omitempty"`
	BundleSHA256        string              `json:"bundle_sha256"`
	InputSHA256         string              `json:"input_sha256"`
	Alert               bool                `json:"alert"`
	Driver              string              `json:"driver"`
	Probability         map[string]float64  `json:"probability"`
	ReasonCodes         map[string][]string `json:"reason_codes"`
	Action              string              `json:"action"`
	ActionNote          string              `json:"action_note"`
	AdverseActionNotice bool                `json:"adverse_action_notice"`
	Features            map[string]*float64 `json:"features,omitempty"`
	PrevHash            string              `json:"prev_hash"`
	Hash                string              `json:"hash"`
}

func hashRecord(r Record) (string, error) {
	r.Hash = ""
	raw, err := json.Marshal(r)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

type Log struct {
	mu   sync.Mutex
	w    io.Writer
	f    *os.File
	seq  uint64
	prev string
	now  func() time.Time
}

// Open appends to path, continuing the chain if the file already has records.
// The path "-" writes to standard output.
func Open(path string) (*Log, error) {
	l := &Log{prev: genesis, now: time.Now}
	if path == "-" {
		l.w = os.Stdout
		return l, nil
	}
	if last, err := lastRecord(path); err != nil {
		return nil, err
	} else if last != nil {
		l.seq, l.prev = last.Seq, last.Hash
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, err
	}
	l.f, l.w = f, f
	return l, nil
}

func lastRecord(path string) (*Record, error) {
	f, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var last *Record
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return nil, fmt.Errorf("audit log %s is not valid JSON lines: %w", path, err)
		}
		last = &r
	}
	return last, sc.Err()
}

// Append completes the record (sequence, time, chain hashes) and writes it.
func (l *Log) Append(r Record) (Record, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	r.Seq = l.seq + 1
	r.Time = l.now().UTC().Format(time.RFC3339Nano)
	r.PrevHash = l.prev
	h, err := hashRecord(r)
	if err != nil {
		return r, err
	}
	r.Hash = h
	if r.DecisionID == "" {
		r.DecisionID = fmt.Sprintf("%d-%s", r.Seq, h[:12])
	}
	raw, err := json.Marshal(r)
	if err != nil {
		return r, err
	}
	if _, err := l.w.Write(append(raw, '\n')); err != nil {
		return r, err
	}
	l.seq, l.prev = r.Seq, r.Hash
	return r, nil
}

// Head returns the sequence number and hash of the last record written.
func (l *Log) Head() (uint64, string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.seq, l.prev
}

func (l *Log) Close() error {
	if l.f != nil {
		return l.f.Close()
	}
	return nil
}

// Verify walks a log and returns the number of valid records and the hash of
// the last one, or the first record at which the chain is broken.
func Verify(path string) (int, string, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, "", err
	}
	defer f.Close()
	prev, n := genesis, 0
	var seq uint64
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var r Record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			return n, prev, fmt.Errorf("record %d: %w", n+1, err)
		}
		if r.Seq != seq+1 {
			return n, prev, fmt.Errorf("record %d: sequence %d follows %d (a record is missing or reordered)", n+1, r.Seq, seq)
		}
		if r.PrevHash != prev {
			return n, prev, fmt.Errorf("record %d (seq %d): prev_hash does not match the preceding record", n+1, r.Seq)
		}
		// decision_id is derived from the hash, so it is excluded when it has that form
		check := r
		if check.DecisionID == fmt.Sprintf("%d-%s", r.Seq, r.Hash[:min(12, len(r.Hash))]) {
			check.DecisionID = ""
		}
		h, err := hashRecord(check)
		if err != nil {
			return n, prev, err
		}
		if h != r.Hash {
			return n, prev, fmt.Errorf("record %d (seq %d): content does not match its hash (edited after writing)", n+1, r.Seq)
		}
		prev, seq = r.Hash, r.Seq
		n++
	}
	return n, prev, sc.Err()
}
