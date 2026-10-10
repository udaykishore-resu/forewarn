// Command forewarn scores accounts with a Forewarn model bundle.
//
//	forewarn serve         HTTP scoring service
//	forewarn batch         score a JSON-lines file
//	forewarn verify        replay reference cases and fail on any difference
//	forewarn inspect       print the model as readable tables
//	forewarn audit-verify  check the hash chain of an audit log
//
// The binary uses only the Go standard library and opens no outbound
// connections.
package main

import (
	"bufio"
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/udaykishore-resu/forewarn/scorer/internal/audit"
	"github.com/udaykishore-resu/forewarn/scorer/internal/batch"
	"github.com/udaykishore-resu/forewarn/scorer/internal/bundle"
	"github.com/udaykishore-resu/forewarn/scorer/internal/drift"
	"github.com/udaykishore-resu/forewarn/scorer/internal/engine"
	"github.com/udaykishore-resu/forewarn/scorer/internal/server"
)

var version = "dev"

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "serve":
		err = cmdServe(os.Args[2:])
	case "batch":
		err = cmdBatch(os.Args[2:])
	case "verify":
		err = cmdVerify(os.Args[2:])
	case "inspect":
		err = cmdInspect(os.Args[2:])
	case "audit-verify":
		err = cmdAuditVerify(os.Args[2:])
	case "version":
		fmt.Println(version)
	case "-h", "--help", "help":
		usage()
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "forewarn:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, `usage: forewarn <command> [flags]

commands:
  serve          run the HTTP scoring service
  batch          score a JSON-lines file of accounts
  verify         replay reference cases against a bundle
  inspect        print a bundle as readable tables
  audit-verify   check the hash chain of an audit log
  version        print the build version

run "forewarn <command> -h" for flags`)
}

type common struct {
	bundlePath      string
	bundleSHA       string
	allowRestricted bool
}

func (c *common) register(fs *flag.FlagSet) {
	fs.StringVar(&c.bundlePath, "bundle", envOr("FOREWARN_BUNDLE", "bundle.json"), "path to the model bundle")
	fs.StringVar(&c.bundleSHA, "bundle-sha256", os.Getenv("FOREWARN_BUNDLE_SHA256"), "expected SHA-256 of the bundle; start fails on mismatch")
	fs.BoolVar(&c.allowRestricted, "allow-restricted", false, "accept bundles that use restricted features")
}

func (c *common) load() (*engine.Engine, error) {
	b, err := bundle.Load(c.bundlePath, bundle.Options{ExpectSHA256: c.bundleSHA, AllowRestricted: c.allowRestricted})
	if err != nil {
		return nil, err
	}
	return engine.New(b)
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func openAudit(path string) (*audit.Log, error) {
	if path == "off" {
		return nil, nil
	}
	return audit.Open(path)
}

// ------------------------------------------------------------------ serve
func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	var c common
	c.register(fs)
	listen := fs.String("listen", envOr("FOREWARN_LISTEN", ":8080"), "address to listen on")
	auditPath := fs.String("audit-log", envOr("FOREWARN_AUDIT_LOG", "-"), `audit log path, "-" for stdout, "off" to disable`)
	auditFeatures := fs.Bool("audit-include-features", false, "write feature values into audit records")
	cert := fs.String("tls-cert", os.Getenv("FOREWARN_TLS_CERT"), "TLS certificate (enables HTTPS)")
	key := fs.String("tls-key", os.Getenv("FOREWARN_TLS_KEY"), "TLS private key")
	clientCA := fs.String("client-ca", os.Getenv("FOREWARN_CLIENT_CA"), "CA bundle; when set, clients must present a certificate")
	fs.Parse(args)

	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))
	eng, err := c.load()
	if err != nil {
		return err
	}
	al, err := openAudit(*auditPath)
	if err != nil {
		return err
	}
	if al != nil {
		defer al.Close()
	}
	mon := drift.NewMonitor(eng.Bundle())
	srv := server.New(eng, al, mon, log, server.Options{AuditIncludeFeatures: *auditFeatures})
	hs := &http.Server{
		Addr:              *listen,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	if *clientCA != "" {
		pem, err := os.ReadFile(*clientCA)
		if err != nil {
			return err
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(pem) {
			return errors.New("client-ca contains no certificates")
		}
		hs.TLSConfig = &tls.Config{ClientCAs: pool, ClientAuth: tls.RequireAndVerifyClientCert, MinVersion: tls.VersionTLS12}
	}
	b := eng.Bundle()
	log.Info("forewarn scorer starting", "listen", *listen, "version", version, "model_version", b.Version,
		"bundle_sha256", b.SHA256, "data_source", b.DataSource, "features", len(b.Features), "tls", *cert != "", "audit", *auditPath)
	if b.DataSource == "synthetic" {
		log.Warn("this bundle was trained on synthetic data and must not be used for decisions about real customers")
	}

	errc := make(chan error, 1)
	go func() {
		if *cert != "" {
			errc <- hs.ListenAndServeTLS(*cert, *key)
		} else {
			errc <- hs.ListenAndServe()
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	select {
	case err := <-errc:
		return err
	case <-stop:
	}
	srv.SetReady(false)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	log.Info("shutting down")
	return hs.Shutdown(ctx)
}

// ------------------------------------------------------------------ batch
func cmdBatch(args []string) error {
	fs := flag.NewFlagSet("batch", flag.ExitOnError)
	var c common
	c.register(fs)
	inPath := fs.String("in", "-", `input JSON lines, "-" for stdin`)
	outPath := fs.String("out", "-", `output JSON lines, "-" for stdout`)
	auditPath := fs.String("audit-log", "off", `audit log path, "-" for stdout, "off" to disable`)
	summaryPath := fs.String("summary", "", "write a run summary with the drift report to this file")
	failOnDrift := fs.Bool("fail-on-drift", false, "exit with status 3 when a score's PSI reaches the alert level")
	failOnFeatureDrift := fs.Bool("fail-on-feature-drift", false, "also exit with status 3 when an input's PSI reaches the alert level")
	fs.Parse(args)

	eng, err := c.load()
	if err != nil {
		return err
	}
	var in io.Reader = os.Stdin
	if *inPath != "-" {
		f, err := os.Open(*inPath)
		if err != nil {
			return err
		}
		defer f.Close()
		in = f
	}
	var out io.Writer = os.Stdout
	if *outPath != "-" {
		f, err := os.Create(*outPath)
		if err != nil {
			return err
		}
		defer f.Close()
		out = f
	}
	al, err := openAudit(*auditPath)
	if err != nil {
		return err
	}
	if al != nil {
		defer al.Close()
	}
	mon := drift.NewMonitor(eng.Bundle())
	srv := server.New(eng, al, mon, slog.New(slog.NewJSONHandler(os.Stderr, nil)), server.Options{})
	sum, err := batch.Run(srv, mon, eng.Bundle().SHA256, in, out)
	if err != nil {
		return err
	}
	if al != nil {
		sum.AuditRecords, sum.AuditHead = al.Head()
	}
	raw, _ := json.MarshalIndent(sum, "", "  ")
	if *summaryPath != "" {
		if err := os.WriteFile(*summaryPath, append(raw, '\n'), 0o644); err != nil {
			return err
		}
	}
	fmt.Fprintf(os.Stderr, "scored %d accounts, %d alerts, score drift %s, input drift %s\n",
		sum.Accounts, sum.Alerts["any"], sum.Drift.ScoreStatus, sum.Drift.FeatureStatus)
	if (*failOnDrift && sum.Drift.ScoreStatus == "alert") || (*failOnFeatureDrift && sum.Drift.FeatureStatus == "alert") {
		os.Exit(3)
	}
	return nil
}

// ------------------------------------------------------------------ verify
type parityCase struct {
	Features map[string]*float64 `json:"features"`
	Expect   engine.Decision     `json:"expect"`
}

type verifyReport struct {
	BundleSHA256 string   `json:"bundle_sha256"`
	Cases        int      `json:"cases"`
	Passed       int      `json:"passed"`
	Failed       int      `json:"failed"`
	MaxDiffLogit float64  `json:"max_abs_diff_logit"`
	MaxDiffProb  float64  `json:"max_abs_diff_probability"`
	MaxDiffGain  float64  `json:"max_abs_diff_gain"`
	Tolerance    float64  `json:"tolerance"`
	Failures     []string `json:"first_failures,omitempty"`
	GoVersion    string   `json:"scorer_version"`
}

func cmdVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ExitOnError)
	var c common
	c.register(fs)
	cases := fs.String("cases", "parity_cases.jsonl", "reference cases exported by the training pipeline")
	tol := fs.Float64("tolerance", 1e-9, "largest accepted absolute difference on any number")
	report := fs.String("report", "", "write the result as JSON to this file")
	fs.Parse(args)

	eng, err := c.load()
	if err != nil {
		return err
	}
	f, err := os.Open(*cases)
	if err != nil {
		return err
	}
	defer f.Close()
	rep := verifyReport{BundleSHA256: eng.Bundle().SHA256, Tolerance: *tol, GoVersion: version}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var pc parityCase
		if err := json.Unmarshal(sc.Bytes(), &pc); err != nil {
			return fmt.Errorf("case %d: %w", rep.Cases+1, err)
		}
		rep.Cases++
		in, err := eng.Vector(pc.Features)
		if err != nil {
			return fmt.Errorf("case %d: %w", rep.Cases, err)
		}
		if msg := compare(pc.Expect, eng.Decide(in), *tol, &rep); msg != "" {
			rep.Failed++
			if len(rep.Failures) < 10 {
				rep.Failures = append(rep.Failures, fmt.Sprintf("case %d: %s", rep.Cases, msg))
			}
		} else {
			rep.Passed++
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	raw, _ := json.MarshalIndent(rep, "", "  ")
	fmt.Println(string(raw))
	if *report != "" {
		if err := os.WriteFile(*report, append(raw, '\n'), 0o644); err != nil {
			return err
		}
	}
	if rep.Failed > 0 || rep.Cases == 0 {
		return fmt.Errorf("%d of %d cases differ from the reference implementation", rep.Failed, rep.Cases)
	}
	return nil
}

func compare(want, got engine.Decision, tol float64, rep *verifyReport) string {
	if want.Alert != got.Alert || want.Driver != got.Driver {
		return fmt.Sprintf("alert/driver: want %v/%q, got %v/%q", want.Alert, want.Driver, got.Alert, got.Driver)
	}
	for name, w := range want.Scores {
		g, ok := got.Scores[name]
		if !ok {
			return "missing score " + name
		}
		dl, dp := math.Abs(w.Logit-g.Logit), math.Abs(w.Probability-g.Probability)
		rep.MaxDiffLogit, rep.MaxDiffProb = math.Max(rep.MaxDiffLogit, dl), math.Max(rep.MaxDiffProb, dp)
		if dl > tol || dp > tol || w.Alert != g.Alert {
			return fmt.Sprintf("%s score: want logit %v alert %v, got %v %v", name, w.Logit, w.Alert, g.Logit, g.Alert)
		}
		if len(w.Reasons) != len(g.Reasons) {
			return fmt.Sprintf("%s reasons: want %d, got %d", name, len(w.Reasons), len(g.Reasons))
		}
		for i := range w.Reasons {
			if w.Reasons[i].Code != g.Reasons[i].Code || w.Reasons[i].Feature != g.Reasons[i].Feature ||
				w.Reasons[i].Text != g.Reasons[i].Text || math.Abs(w.Reasons[i].Points-g.Reasons[i].Points) > tol {
				return fmt.Sprintf("%s reason %d: want %s (%v), got %s (%v)", name, i+1, w.Reasons[i].Code, w.Reasons[i].Points, g.Reasons[i].Code, g.Reasons[i].Points)
			}
		}
	}
	wa, ga := want.Action, got.Action
	if wa.Recommended != ga.Recommended || wa.Note != ga.Note || wa.InPilotSupport != ga.InPilotSupport || wa.AdverseActionNotice != ga.AdverseActionNotice {
		return fmt.Sprintf("action: want %s (%s), got %s (%s)", wa.Recommended, wa.Note, ga.Recommended, ga.Note)
	}
	for arm, w := range wa.EstimatedGain {
		d := math.Abs(w - ga.EstimatedGain[arm])
		rep.MaxDiffGain = math.Max(rep.MaxDiffGain, d)
		if d > tol*math.Max(1, math.Abs(w)) {
			return fmt.Sprintf("gain for %s: want %v, got %v", arm, w, ga.EstimatedGain[arm])
		}
		if wa.Eligible[arm] != ga.Eligible[arm] {
			return fmt.Sprintf("eligibility for %s: want %v, got %v", arm, wa.Eligible[arm], ga.Eligible[arm])
		}
	}
	return ""
}

// ------------------------------------------------------------------ inspect
func cmdInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ExitOnError)
	var c common
	c.register(fs)
	fs.Parse(args)
	b, err := bundle.Load(c.bundlePath, bundle.Options{ExpectSHA256: c.bundleSHA, AllowRestricted: c.allowRestricted})
	if err != nil {
		return err
	}
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	fmt.Fprintf(w, "bundle   %s %s\nsha256   %s\ncreated  %s\nsource   %s\n", b.Name, b.Version, b.SHA256, b.CreatedAt, b.DataSource)
	dir := map[string]int{}
	for _, f := range b.Features {
		dir[f.Name] = f.Monotone
	}
	show := func(title string, m bundle.Model) {
		fmt.Fprintf(w, "\n== %s ==\nscore = %.6f", title, m.Intercept)
		for _, t := range m.Terms {
			fmt.Fprintf(w, " + f(%s)", t.Feature)
		}
		if m.Link == "logit" {
			fmt.Fprintf(w, "\nprobability = sigmoid(%.6f + %.6f * score)\n", m.Calibration.A, m.Calibration.B)
		} else {
			fmt.Fprintln(w)
		}
		for _, t := range m.Terms {
			arrow := map[int]string{1: "risk rises with value", -1: "risk falls with value", 0: "unconstrained"}[dir[t.Feature]]
			fmt.Fprintf(w, "\n  %s  (%s)\n", t.Feature, arrow)
			for i, v := range t.Values {
				lo, hi := "-inf", "+inf"
				if i > 0 {
					lo = fmt.Sprintf("%.6g", t.Thresholds[i-1])
				}
				if i < len(t.Thresholds) {
					hi = fmt.Sprintf("%.6g", t.Thresholds[i])
				}
				fmt.Fprintf(w, "    (%12s, %12s]  %+.5f\n", lo, hi, v)
			}
			fmt.Fprintf(w, "    %28s  %+.5f\n", "missing", t.Missing)
		}
	}
	show("delinquency", b.Models.Delinquency)
	fmt.Fprintf(w, "\nalert when probability >= %.6f\n", b.Alerts["delinquency"].ThresholdProbability)
	show("bankruptcy", b.Models.Bankruptcy)
	fmt.Fprintf(w, "\nalert when probability >= %.6f\n", b.Alerts["bankruptcy"].ThresholdProbability)
	for _, arm := range b.Policy.Arms {
		show("value effect: "+arm, b.Models.ValueEffects[arm])
	}
	return nil
}

// ------------------------------------------------------------------ audit-verify
func cmdAuditVerify(args []string) error {
	fs := flag.NewFlagSet("audit-verify", flag.ExitOnError)
	path := fs.String("file", "", "audit log to check")
	expectHead := fs.String("expect-head", "", "hash of the last record, as recorded elsewhere (for example audit_head in a batch summary)")
	expectRecords := fs.Int("expect-records", -1, "number of records the log should hold")
	fs.Parse(args)
	if *path == "" {
		return errors.New("-file is required")
	}
	n, head, err := audit.Verify(*path)
	if err != nil {
		return fmt.Errorf("chain broken after %d valid records: %w", n, err)
	}
	if *expectRecords >= 0 && n != *expectRecords {
		return fmt.Errorf("log holds %d records, expected %d (records were removed or added at the end)", n, *expectRecords)
	}
	if *expectHead != "" && head != *expectHead {
		return fmt.Errorf("head is %s, expected %s (the end of the log is not what was recorded)", head, *expectHead)
	}
	fmt.Printf("ok: %d records, chain intact, head %s\n", n, head)
	if *expectHead == "" && *expectRecords < 0 {
		fmt.Println("note: without -expect-head or -expect-records, records removed from the end go undetected")
	}
	return nil
}
