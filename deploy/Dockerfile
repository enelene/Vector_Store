# syntax=docker/dockerfile:1

# ──────────────────────────────────────────────────────────────────────────────
# VectorDB — production container image (multi-stage build)
#
# Stage 1 compiles a fully static Go binary; stage 2 copies only that binary
# into a distroless base, producing a tiny, shell-free, non-root final image.
#
# IMPORTANT: build from the REPOSITORY ROOT so the whole module is in context:
#
#   docker build -f deploy/Dockerfile -t vectordb:latest .
# ──────────────────────────────────────────────────────────────────────────────

# ---- Stage 1: build ----------------------------------------------------------
# Pin the builder to the Go version declared in go.mod (go 1.25).
FROM golang:1.25 AS builder

WORKDIR /src

# Download dependencies first. This layer is cached and only re-runs when
# go.mod / go.sum change, which keeps incremental builds fast.
COPY go.mod go.sum ./
RUN go mod download

# Copy the remaining source and compile.
COPY . .

# CGO_ENABLED=0 yields a static binary with no libc dependency — required to run
# on the distroless "static" base. -trimpath removes local filesystem paths and
# -ldflags "-s -w" strips the symbol and DWARF debug tables to shrink the binary.
RUN CGO_ENABLED=0 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /out/vectordb-server \
        ./cmd/server

# ---- Stage 2: runtime --------------------------------------------------------
# distroless "static" ships only CA certificates, tzdata and a non-root user —
# no shell and no package manager — minimising both image size and attack surface.
FROM gcr.io/distroless/static-debian12:nonroot

# Copy just the compiled binary out of the builder stage.
COPY --from=builder /out/vectordb-server /vectordb-server

# gRPC port — the client API and internal Raft RPCs share this single port.
EXPOSE 50051

# Raft state snapshot lives here; back it with a PersistentVolume in Kubernetes.
VOLUME ["/data"]

# Run as the built-in non-root user (uid 65532) for least privilege.
USER nonroot:nonroot

ENTRYPOINT ["/vectordb-server"]
