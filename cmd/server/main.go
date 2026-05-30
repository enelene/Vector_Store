// main is the entry point for a single VectorDB cluster node.
//
// Each node exposes one gRPC port that serves both the client-facing
// VectorDatabase service and the internal RaftService used for consensus.
//
// Configuration can come from CLI flags (convenient locally) or from
// environment variables (convenient in Kubernetes, where every pod in a
// StatefulSet runs an identical spec).  Precedence is: flag → env → default.
//
// Local usage:
//
//	vectordb_server \
//	  --id=0 \
//	  --port=50051 \
//	  --peers=localhost:50051,localhost:50052,localhost:50053 \
//	  --data-dir=./data0
//
//	--id        Index of this node in --peers (0-based). -1 derives it from the
//	            hostname ordinal (e.g. "vectordb-2" → 2), which is how nodes get
//	            their identity inside a StatefulSet.
//	--port      TCP port to listen on.
//	--peers     Comma-separated list of ALL node addresses (including self).
//	--data-dir  Directory for the Raft state snapshot; empty disables persistence.
//
// Kubernetes usage (see deploy/k8s): set VECTORDB_PEERS, VECTORDB_PORT,
// VECTORDB_DATA_DIR and inject POD_NAME via the downward API.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof" // registers /debug/pprof/* handlers on http.DefaultServeMux
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	pb "vectordb/api/proto"
	"vectordb/internal/consensus"
	"vectordb/internal/network"
	"vectordb/internal/sharding"
	"vectordb/internal/vectorindex"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
)

// shutdownGrace bounds how long we wait for in-flight RPCs to drain during a
// graceful stop.  It is kept below the Kubernetes terminationGracePeriodSeconds
// (30s in our manifest) so the node always steps down and snapshots its state
// before the SIGKILL fallback fires.
const shutdownGrace = 10 * time.Second

// nodeConfig is the fully-resolved runtime configuration for this node.
type nodeConfig struct {
	id        int      // index of this node within peers
	port      string   // gRPC listen port
	peers     []string // addresses of ALL nodes, including self
	dataDir   string   // directory for the Raft state snapshot ("" = no persistence)
	pprofAddr string   // address for the net/http/pprof server ("" = disabled)
}

func main() {
	cfg := resolveConfig()

	if cfg.id < 0 || cfg.id >= len(cfg.peers) {
		log.Fatalf("resolved node id %d is out of range for %d peers %v",
			cfg.id, len(cfg.peers), cfg.peers)
	}
	selfAddr := cfg.peers[cfg.id]

	log.Printf("starting node %d  self=%s  peers=%v  dataDir=%q",
		cfg.id, selfAddr, cfg.peers, cfg.dataDir)

	// Profiling endpoint (CPU/heap/goroutine/allocs) for load testing.
	startPprof(cfg.pprofAddr)

	// ── HNSW graph ─────────────────────────────────────────────────────────
	// The graph starts empty.  The Applier populates it as Raft commits Insert
	// commands — both fresh writes and entries replayed from a restored snapshot.
	graph := vectorindex.NewHNSWGraph(
		vectorindex.DefaultM,
		vectorindex.DefaultEfConstruction,
	)

	// ── Consistent hash ring ───────────────────────────────────────────────
	ring := sharding.NewHashRing(0) // 0 → use default 150 virtual nodes
	for _, p := range cfg.peers {
		ring.AddNode(p)
	}

	// ── Raft consensus ─────────────────────────────────────────────────────
	applyCh := make(chan consensus.ApplyMsg, 64)
	notifier := consensus.NewCommitNotifier()

	statePath := raftStatePath(cfg) // "" disables persistence
	raftNode := consensus.Make(cfg.peers, cfg.id, applyCh, statePath)

	// ── Applier ────────────────────────────────────────────────────────────
	// Runs for the lifetime of the process, consuming committed log entries from
	// Raft and inserting them into the local HNSW graph.
	applier := consensus.NewApplier(applyCh, graph, notifier)
	go applier.Run()

	// ── gRPC server ────────────────────────────────────────────────────────
	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", cfg.port))
	if err != nil {
		log.Fatalf("failed to listen on port %s: %v", cfg.port, err)
	}

	grpcServer := grpc.NewServer()

	// Client-facing VectorDatabase service.
	vectorSrv := network.NewVectorDBServer(graph, raftNode, ring, selfAddr, notifier)
	pb.RegisterVectorDatabaseServer(grpcServer, vectorSrv)

	// Internal Raft consensus service (peer-to-peer only).
	pb.RegisterRaftServiceServer(grpcServer, raftNode)

	// Server reflection lets grpcurl introspect services without a .proto file.
	reflection.Register(grpcServer)

	log.Printf("node %d listening on :%s", cfg.id, cfg.port)

	// ── Serve ──────────────────────────────────────────────────────────────
	errCh := make(chan error, 1)
	go func() { errCh <- grpcServer.Serve(lis) }()

	// Kubernetes sends SIGTERM before deleting a pod; Ctrl+C sends SIGINT.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		log.Printf("node %d received %s — beginning graceful shutdown", cfg.id, sig)
		gracefulShutdown(grpcServer, raftNode, cfg.id)
	case err := <-errCh:
		log.Fatalf("node %d gRPC error: %v", cfg.id, err)
	}

	log.Printf("node %d stopped cleanly", cfg.id)
}

// ── Configuration ───────────────────────────────────────────────────────────

// resolveConfig merges command-line flags with environment variables so the
// same binary runs both locally (flags) and in Kubernetes (env + downward API).
//
// Precedence (highest first): explicit flag → environment variable → default.
func resolveConfig() nodeConfig {
	idFlag := flag.Int("id", -1, "node index in --peers; -1 derives it from the hostname ordinal")
	portFlag := flag.String("port", "50051", "gRPC listen port")
	peersFlag := flag.String("peers", "localhost:50051", "comma-separated addresses of ALL nodes")
	dataDirFlag := flag.String("data-dir", "", "directory for the Raft state snapshot; empty disables persistence")
	pprofFlag := flag.String("pprof-addr", "localhost:6060", "address for the net/http/pprof server; empty disables it")
	flag.Parse()

	cfg := nodeConfig{
		port:      envOr("VECTORDB_PORT", *portFlag),
		dataDir:   envOr("VECTORDB_DATA_DIR", *dataDirFlag),
		pprofAddr: envOr("VECTORDB_PPROF_ADDR", *pprofFlag),
	}

	// Peers: the env list (Kubernetes) wins over the flag (local).
	if envPeers := os.Getenv("VECTORDB_PEERS"); envPeers != "" {
		cfg.peers = strings.Split(envPeers, ",")
	} else {
		cfg.peers = strings.Split(*peersFlag, ",")
	}

	// Node id: explicit flag wins; otherwise VECTORDB_ID; otherwise the ordinal
	// suffix of the pod hostname (the StatefulSet's stable identity).
	switch {
	case *idFlag >= 0:
		cfg.id = *idFlag
	case os.Getenv("VECTORDB_ID") != "":
		v, err := strconv.Atoi(os.Getenv("VECTORDB_ID"))
		if err != nil {
			log.Fatalf("invalid VECTORDB_ID %q: %v", os.Getenv("VECTORDB_ID"), err)
		}
		cfg.id = v
	default:
		cfg.id = ordinalFromHostname()
	}

	return cfg
}

// envOr returns the value of environment variable key, or def if unset/empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// ordinalFromHostname extracts the trailing integer of the pod hostname, which
// Kubernetes sets to "<statefulset>-<ordinal>" (e.g. "vectordb-0" → 0).  Falls
// back to 0 when no ordinal is present (e.g. a plain local run).
func ordinalFromHostname() int {
	host := os.Getenv("POD_NAME")
	if host == "" {
		host, _ = os.Hostname()
	}
	if i := strings.LastIndex(host, "-"); i >= 0 {
		if n, err := strconv.Atoi(host[i+1:]); err == nil {
			return n
		}
	}
	log.Printf("could not derive node id from hostname %q; defaulting to 0", host)
	return 0
}

// raftStatePath returns the file path for this node's Raft snapshot, or "" if
// persistence is disabled.  Each pod owns its own PersistentVolume, so a single
// fixed filename inside the data dir is sufficient.
func raftStatePath(cfg nodeConfig) string {
	if cfg.dataDir == "" {
		return ""
	}
	if err := os.MkdirAll(cfg.dataDir, 0o755); err != nil {
		log.Printf("could not create data dir %q: %v (persistence disabled)", cfg.dataDir, err)
		return ""
	}
	return filepath.Join(cfg.dataDir, "raft-state.json")
}

// startPprof launches the net/http/pprof profiling server on addr (if non-empty)
// in a background goroutine. It serves the standard /debug/pprof/* endpoints
// (heap, profile, goroutine, allocs, …) registered by the blank import.
//
// It defaults to localhost so profiles are not exposed off-box; set
// VECTORDB_PPROF_ADDR=":6060" (and a containerPort) to reach it in Kubernetes
// via `kubectl port-forward`.
func startPprof(addr string) {
	if addr == "" {
		return // profiling disabled
	}
	go func() {
		log.Printf("pprof profiling server on http://%s/debug/pprof/", addr)
		if err := http.ListenAndServe(addr, nil); err != nil {
			log.Printf("pprof server stopped: %v", err)
		}
	}()
}

// ── Graceful shutdown ───────────────────────────────────────────────────────

// gracefulShutdown drains the gRPC server, then steps the Raft node down and
// snapshots its state.
//
// Ordering matters:
//   - Client RPCs are drained FIRST (GracefulStop) so writes already accepted by
//     this node can still reach commit while it is briefly still the leader.
//   - THEN raft.Shutdown() steps down and snapshots, after which the surviving
//     nodes elect a new leader within one election timeout.
//
// GracefulStop is bounded by shutdownGrace; if in-flight work does not drain in
// time (e.g. the cluster lost quorum) we force-stop so shutdown still completes
// before Kubernetes escalates to SIGKILL.
func gracefulShutdown(srv *grpc.Server, rf *consensus.Raft, id int) {
	stopped := make(chan struct{})
	go func() {
		srv.GracefulStop()
		close(stopped)
	}()

	select {
	case <-stopped:
		log.Printf("node %d gRPC drained gracefully", id)
	case <-time.After(shutdownGrace):
		log.Printf("node %d gRPC drain timed out after %s — forcing stop", id, shutdownGrace)
		srv.Stop()
	}

	// Step down as leader (if applicable) and snapshot state to disk.
	rf.Shutdown()
}
