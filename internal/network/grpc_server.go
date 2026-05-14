// Package network wires gRPC transport to the HNSW index and Raft consensus.
//
// Insert routing (two-hop maximum):
//
//	Hop 0 — Hash ring routing
//	  If this node is NOT the hash-ring owner of the key, forward the request
//	  to the responsible node using a gRPC client call.  The forwarded request
//	  carries the metadata key "x-vdb-hop: 1" so the receiver skips re-routing.
//
//	Hop 1 — Raft submission
//	  The hash-ring owner submits the command to Raft.  If it is not currently
//	  the Raft leader, it forwards once more to the known leader (still hop 1,
//	  preventing further forwarding loops).
//	  The leader waits for the entry to be committed before responding.
//
// Search is always handled locally: because Raft replicates every Insert to
// every node, all HNSW graphs are eventually identical and any node can answer
// a nearest-neighbour query.
package network

import (
	"context"
	"strconv"
	"sync"
	"time"

	pb "vectordb/api/proto"
	"vectordb/internal/consensus"
	"vectordb/internal/sharding"
	"vectordb/internal/vectorindex"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const (
	// hopMetaKey carries the number of forwards a request has already gone
	// through. A request can legitimately need TWO forwards — one for hash-ring
	// routing (to the responsible node) and one for leader routing (if that node
	// is not the current Raft leader) — so we count hops instead of using a
	// binary "was forwarded" flag.
	hopMetaKey = "x-vdb-hop"

	// maxHops bounds total forwards per Insert so a stale leader pointer cannot
	// produce an infinite forwarding loop. 2 covers hash → leader; anything more
	// gives up and returns Unavailable so the client can retry.
	maxHops = 2

	// commitTimeout is the upper bound for waiting on Raft to commit an entry.
	// Context deadlines passed by the caller take precedence if shorter.
	commitTimeout = 5 * time.Second
)

// VectorDBServer implements pb.VectorDatabaseServer with distributed routing.
type VectorDBServer struct {
	pb.UnimplementedVectorDatabaseServer

	index    *vectorindex.HNSWGraph
	raft     *consensus.Raft
	ring     *sharding.HashRing
	self     string // this node's gRPC listen address (e.g. "localhost:50051")
	notifier *consensus.CommitNotifier

	// Cached gRPC clients for forwarding to peer VectorDatabase endpoints.
	peerMu      sync.Mutex
	peerClients map[string]pb.VectorDatabaseClient
}

// NewVectorDBServer constructs the server.  graph, raft, ring, and notifier
// must already be initialised; this constructor does not start any goroutines.
func NewVectorDBServer(
	graph *vectorindex.HNSWGraph,
	raft *consensus.Raft,
	ring *sharding.HashRing,
	self string,
	notifier *consensus.CommitNotifier,
) *VectorDBServer {
	return &VectorDBServer{
		index:       graph,
		raft:        raft,
		ring:        ring,
		self:        self,
		notifier:    notifier,
		peerClients: make(map[string]pb.VectorDatabaseClient),
	}
}

// ── Insert ─────────────────────────────────────────────────────────────────────

// Insert routes the write through the hash ring and Raft consensus layer.
func (s *VectorDBServer) Insert(
	ctx context.Context, req *pb.InsertRequest,
) (*pb.InsertResponse, error) {
	if req.Id == "" {
		return nil, status.Error(codes.InvalidArgument, "id must not be empty")
	}
	if len(req.Vector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vector must not be empty")
	}

	hops := incomingHops(ctx)

	// ── Hop 0: hash-ring routing ───────────────────────────────────────────
	// Only apply on the FIRST entry into the system (hops == 0). After any
	// forward we trust the upstream caller's routing decision and go directly
	// to Raft, so the request can't bounce back through the hash ring.
	if hops == 0 {
		responsible := s.ring.GetResponsibleNode(req.Id)
		if responsible != "" && responsible != s.self {
			return s.forwardInsert(ctx, responsible, req, hops)
		}
	}

	// ── Hop 1: Raft submission ─────────────────────────────────────────────
	vec := make([]float32, len(req.Vector))
	copy(vec, req.Vector)
	cmd := consensus.InsertCommand{ID: req.Id, Vector: vec}

	logIdx, _, isLeader := s.raft.Start(cmd)
	if !isLeader {
		// This node is not the Raft leader. Forward to the known leader, as long
		// as we haven't already used both hops — which guarantees at most one
		// hash-routing forward + one leader forward = two network hops total.
		leaderAddr := s.raft.GetLeaderAddress()
		if hops < maxHops && leaderAddr != "" && leaderAddr != s.self {
			return s.forwardInsert(ctx, leaderAddr, req, hops)
		}
		return nil, status.Error(codes.Unavailable,
			"no Raft leader available — retry in a moment")
	}

	// ── Wait for the entry to be committed and applied ────────────────────
	// Register a channel that will be closed by the Applier once index logIdx
	// has been applied to the state machine.
	notify := s.notifier.Register(logIdx)

	// Honour the caller's deadline, but cap at commitTimeout if longer.
	deadline := time.Now().Add(commitTimeout)
	if d, ok := ctx.Deadline(); ok && d.Before(deadline) {
		deadline = d
	}
	timer := time.NewTimer(time.Until(deadline))
	defer timer.Stop()

	select {
	case <-notify:
		return &pb.InsertResponse{Id: req.Id, Success: true}, nil
	case <-timer.C:
		return nil, status.Error(codes.DeadlineExceeded,
			"timed out waiting for Raft commit — the cluster may be degraded")
	case <-ctx.Done():
		return nil, status.FromContextError(ctx.Err()).Err()
	}
}

// ── Search ─────────────────────────────────────────────────────────────────────

// Search is always served locally.  Because Raft replicates every Insert to
// every node the local HNSW graph is a complete replica of the cluster state.
func (s *VectorDBServer) Search(
	_ context.Context, req *pb.SearchRequest,
) (*pb.SearchResponse, error) {
	if len(req.QueryVector) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query_vector must not be empty")
	}
	if req.TopK <= 0 {
		return nil, status.Error(codes.InvalidArgument, "top_k must be > 0")
	}

	results := s.index.Search(req.QueryVector, int(req.TopK))

	pbResults := make([]*pb.SearchResult, len(results))
	for i, r := range results {
		pbResults[i] = &pb.SearchResult{Id: r.ID, Score: r.Score}
	}
	return &pb.SearchResponse{Results: pbResults}, nil
}

// ── Forwarding helpers ──────────────────────────────────────────────────────

// forwardInsert sends the Insert RPC to addr with the hop counter incremented
// by one, so the receiver knows how many forwards this request has already
// undergone and can decide whether further hops are still allowed.
// The caller's deadline is propagated so the total latency budget is respected.
func (s *VectorDBServer) forwardInsert(
	ctx context.Context, addr string, req *pb.InsertRequest, currentHops int,
) (*pb.InsertResponse, error) {
	client, err := s.getPeerClient(addr)
	if err != nil {
		return nil, status.Errorf(codes.Unavailable,
			"cannot reach node %s: %v", addr, err)
	}

	outCtx := metadata.NewOutgoingContext(ctx,
		metadata.Pairs(hopMetaKey, strconv.Itoa(currentHops+1)))

	return client.Insert(outCtx, req)
}

// getPeerClient returns a cached (or freshly dialled) VectorDatabaseClient for
// addr.  Connections are created lazily and cached for the lifetime of the server.
func (s *VectorDBServer) getPeerClient(addr string) (pb.VectorDatabaseClient, error) {
	s.peerMu.Lock()
	defer s.peerMu.Unlock()

	if c, ok := s.peerClients[addr]; ok {
		return c, nil
	}

	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	c := pb.NewVectorDatabaseClient(conn)
	s.peerClients[addr] = c
	return c, nil
}

// incomingHops returns the number of forwards a request has already gone
// through, read from the gRPC metadata header. Missing / unparseable values
// are treated as 0 (i.e. a fresh, never-forwarded request).
func incomingHops(ctx context.Context) int {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return 0
	}
	vals := md.Get(hopMetaKey)
	if len(vals) == 0 {
		return 0
	}
	n, err := strconv.Atoi(vals[0])
	if err != nil || n < 0 {
		return 0
	}
	return n
}
