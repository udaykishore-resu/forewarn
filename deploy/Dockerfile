# Forewarn scorer image.
#
# Build context is the repository root:
#   docker build -f deploy/Dockerfile -t forewarn-scorer:0.1.0 .
#
# The image holds one statically linked binary and nothing else: no shell, no
# package manager, no model. The model bundle is mounted at runtime and pinned
# by digest, so the same image is promoted unchanged from test to production.
#
# Inside a bank, replace both base images with the copies in the bank's own
# registry and pin them by digest.
FROM golang:1.24 AS build
WORKDIR /src
COPY scorer/go.mod ./
COPY scorer/ ./
ARG VERSION=0.1.0
# No module downloads happen: the scorer has no third-party dependencies.
RUN CGO_ENABLED=0 go test ./... \
 && CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /out/forewarn ./cmd/forewarn

FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/forewarn /forewarn
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/forewarn"]
CMD ["serve", "-bundle", "/bundle/bundle.json", "-listen", ":8080", "-audit-log", "-"]
