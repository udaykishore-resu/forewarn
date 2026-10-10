# Forewarn: build, train, validate, verify.
#
#   make setup     install Python dependencies
#   make demo      whole pipeline on 20,000 simulated customers (about a minute)
#   make all       whole pipeline on 100,000 simulated customers (about 15 minutes, 5 GB of memory)
#   make test      Python and Go tests
#   make verify    Go scorer against the Python reference cases
#
# macOS: LightGBM needs OpenMP (`brew install libomp`).

PY        ?= python3
CUSTOMERS ?= 100000
BUNDLE    := artifacts/bundle/bundle.json
CASES     := artifacts/bundle/parity_cases.jsonl
VERSION   ?= 0.1.0

.PHONY: setup demo all pipeline build verify report test test-py test-go testdata no-egress-check serve inspect image clean

setup:
	$(PY) -m pip install -r requirements.txt

demo:
	$(MAKE) all CUSTOMERS=20000

all: pipeline build verify report

pipeline:
	cd ml && $(PY) -m forewarn.cli simulate --customers $(CUSTOMERS)
	cd ml && $(PY) -m forewarn.cli features
	cd ml && $(PY) -m forewarn.cli train
	cd ml && $(PY) -m forewarn.cli recommend
	cd ml && $(PY) -m forewarn.cli export
	cd ml && $(PY) -m forewarn.cli validate

build: no-egress-check
	cd scorer && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=$(VERSION)" -o ../bin/forewarn ./cmd/forewarn

# Replays the reference cases through the Go scorer and records the result for the report.
verify: build
	./bin/forewarn verify -bundle $(BUNDLE) -bundle-sha256 $$(cat artifacts/bundle/bundle.sha256) -cases $(CASES) -report artifacts/metrics/parity.json

report:
	cd ml && $(PY) -m forewarn.cli report

test: test-py test-go

test-py:
	cd ml && $(PY) -m pytest -q

test-go:
	cd scorer && go vet ./... && go test ./...

# Copies the current bundle and a sample of reference cases next to the Go tests,
# so `go test` checks parity without the Python pipeline.
testdata:
	mkdir -p scorer/internal/engine/testdata
	cp $(BUNDLE) scorer/internal/engine/testdata/bundle.json
	head -n 120 $(CASES) > scorer/internal/engine/testdata/parity_cases.jsonl

# The scorer must contain no outbound network client.
no-egress-check:
	@if grep -rnE 'http\.(Get|Post|PostForm|Head|NewRequest|NewRequestWithContext|Client|DefaultClient|Transport)|net\.Dial|net\.Resolver|rpc\.Dial|smtp\.|exec\.Command' \
	    --include='*.go' --exclude='*_test.go' scorer; then \
	  echo "outbound client or subprocess found in scorer source"; exit 1; \
	else echo "no-egress-check: ok"; fi

serve: build
	./bin/forewarn serve -bundle $(BUNDLE) -bundle-sha256 $$(cat artifacts/bundle/bundle.sha256) -listen 127.0.0.1:8080 -audit-log artifacts/audit.jsonl

inspect: build
	./bin/forewarn inspect -bundle $(BUNDLE) | less

image:
	docker build -f deploy/Dockerfile --build-arg VERSION=$(VERSION) -t forewarn-scorer:$(VERSION) .

clean:
	rm -rf data bin artifacts/audit.jsonl
