# ──────────────────────────────────────────────────────────────────────────────
# VectorDB — top-level workflow targets
#
# Quick start (from a fresh clone, with Docker + kind + kubectl + go installed):
#
#   make image            # build the server image
#   make up               # create kind cluster, load image, deploy 3-node StatefulSet
#   make demo             # insert sample vectors, then search them
#   make down             # tear it all down
#
# Common day-to-day:
#
#   make logs             # tail logs from all three pods
#   make leader           # which pod is the current Raft leader?
#   make benchmark        # port-forward + run ./cmd/benchmarker
#   make resilience       # kill the leader, verify recovery
#   make plots            # regenerate docs/figures/*.png from benchmark_results.csv
#   make example          # run the embedded semantic-search example
#   make test             # go test -race ./...
# ──────────────────────────────────────────────────────────────────────────────

# Variables — override on the command line, e.g. `make up KIND_CLUSTER=foo`
KIND_CLUSTER ?= vectordb
IMAGE        ?= vectordb:latest
NAMESPACE    ?= default
LOCAL_PORT   ?= 50551

# Use bash and fail fast inside each recipe line.
SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

# Always rebuild these targets; they're not real files.
.PHONY: help image kind-up load up down logs leader demo resilience benchmark plots example fmt vet test clean

# Default target.
help:
	@awk 'BEGIN { FS = ":.*?## " } /^[a-zA-Z_-]+:.*## / { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

image: ## Build the production server image (multi-stage, distroless).
	docker build -f deploy/Dockerfile -t $(IMAGE) .

kind-up: ## Create the kind cluster (no-op if it already exists).
	@kind get clusters 2>/dev/null | grep -qx "$(KIND_CLUSTER)" || kind create cluster --name $(KIND_CLUSTER)

load: image kind-up ## Build the image and load it into the kind cluster.
	kind load docker-image $(IMAGE) --name $(KIND_CLUSTER)

up: load ## Apply the manifests and wait for all three pods to be Ready.
	kubectl apply -f deploy/k8s/service.yaml
	kubectl apply -f deploy/k8s/statefulset.yaml
	kubectl rollout status statefulset/vectordb --timeout=120s

down: ## Delete the StatefulSet and all PersistentVolumeClaims (data is wiped).
	-kubectl delete statefulset vectordb
	-kubectl delete pvc -l app=vectordb

logs: ## Tail the last 20 log lines from every pod.
	@for p in vectordb-0 vectordb-1 vectordb-2; do \
	  echo "── $$p ──"; kubectl logs $$p --tail=20 || true; \
	done

leader: ## Print the current Raft leader (parses pod logs).
	@best=-1; leader=""; \
	for p in $$(kubectl get pods -l app=vectordb -o jsonpath='{.items[*].metadata.name}'); do \
	  line=$$(kubectl logs $$p 2>/dev/null | grep "became leader for term" | tail -n1 || true); \
	  [ -z "$$line" ] && continue; \
	  t=$$(echo "$$line" | sed -E 's/.*term ([0-9]+).*/\1/'); \
	  if [ "$$t" -gt "$$best" ]; then best=$$t; leader=$$p; fi; \
	done; \
	if [ -n "$$leader" ]; then echo "leader: $$leader (term $$best)"; else echo "no leader found yet"; fi

demo: ## Run the narrated happy-path demo (insert + search).
	bash scripts/demo.sh

resilience: ## Run the chaos test (kills the leader, verifies recovery).
	bash scripts/test_resilience.sh

benchmark: ## Port-forward to vectordb-0 and run the benchmarker (n=10000, q=5000).
	@kubectl port-forward pod/vectordb-0 $(LOCAL_PORT):50051 >/dev/null 2>&1 & PF=$$!; \
	  trap "kill $$PF 2>/dev/null" EXIT; sleep 2; \
	  go run ./cmd/benchmarker --addr=localhost:$(LOCAL_PORT) --n=10000 --queries=5000 --topk=10

plots: ## Regenerate docs/figures/*.png from benchmark_results.csv.
	python3 scripts/generate_plots.py

example: ## Run the semantic-search example against a port-forwarded server.
	@kubectl port-forward pod/vectordb-0 $(LOCAL_PORT):50051 >/dev/null 2>&1 & PF=$$!; \
	  trap "kill $$PF 2>/dev/null" EXIT; sleep 2; \
	  go run ./examples/semantic_search --addr=localhost:$(LOCAL_PORT)

fmt: ## gofmt -w on all Go sources.
	gofmt -w cmd internal pkg examples

vet: ## go vet ./...
	go vet ./...

test: ## go test -race -count=1 ./...
	go test -race -count=1 ./...

clean: down ## Delete the StatefulSet, PVCs, and the kind cluster itself.
	-kind delete cluster --name $(KIND_CLUSTER)
