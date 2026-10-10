package audit

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, path string, n int) {
	t.Helper()
	l, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	for i := 0; i < n; i++ {
		if _, err := l.Append(Record{AccountRef: "acct", BundleSHA256: "abc", Action: "none", Probability: map[string]float64{"delinquency": 0.1}}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestChainVerifiesAndContinuesAcrossRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	write(t, path, 3)
	write(t, path, 2) // a second process appends to the same file
	n, head, err := Verify(path)
	if err != nil || n != 5 || len(head) != 64 {
		t.Fatalf("Verify = %d, %q, %v; want 5 valid records and a head hash", n, head, err)
	}
}

// Dropping records from the end leaves a valid, shorter chain. Only a head
// recorded elsewhere reveals it.
func TestTruncationNeedsAnExternalHead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	write(t, path, 4)
	_, fullHead, _ := Verify(path)
	raw, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	os.WriteFile(path, []byte(strings.Join(lines[:3], "\n")+"\n"), 0o600)
	n, head, err := Verify(path)
	if err != nil || n != 3 {
		t.Fatalf("a truncated chain should still verify internally: %d, %v", n, err)
	}
	if head == fullHead {
		t.Fatal("truncation did not change the head hash, so it could not be detected")
	}
}

func TestEditedRecordIsDetected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	write(t, path, 4)
	raw, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	lines[1] = strings.Replace(lines[1], `"action":"none"`, `"action":"limit_decrease"`, 1)
	os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600)
	n, _, err := Verify(path)
	if err == nil || n != 1 || !strings.Contains(err.Error(), "edited") {
		t.Fatalf("Verify = %d, %v; want failure at the second record", n, err)
	}
}

func TestDeletedRecordIsDetected(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	write(t, path, 4)
	raw, _ := os.ReadFile(path)
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	os.WriteFile(path, []byte(strings.Join(append(lines[:1], lines[2:]...), "\n")+"\n"), 0o600)
	if n, _, err := Verify(path); err == nil || n != 1 {
		t.Fatalf("Verify = %d, %v; want failure after the first record", n, err)
	}
}
