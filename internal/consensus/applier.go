// Package consensus — applier.go
//
// Applier is the bridge between the Raft commit log and the HNSW state machine.
// It drains the Raft applyCh, applies each InsertCommand to the local HNSW
// graph, and signals any waiting gRPC handlers via CommitNotifier.
package consensus

import (
	"log"
	"sync"

	"vectordb/internal/vectorindex"
)

// ── CommitNotifier ────────────────────────────────────────────────────────────

// CommitNotifier lets a gRPC handler wait for a specific log index to be
// applied to the state machine before returning a response to the client.
//
// Usage:
//
//	ch := notifier.Register(logIndex)
//	select {
//	case <-ch:           // entry committed and applied
//	case <-ctx.Done():   // caller timed out or cancelled
//	}
type CommitNotifier struct {
	mu  sync.Mutex
	chs map[int]chan struct{}
}

// NewCommitNotifier allocates an empty notifier.
func NewCommitNotifier() *CommitNotifier {
	return &CommitNotifier{chs: make(map[int]chan struct{})}
}

// Register creates and returns a channel that will be closed when the entry
// at logIndex is applied.  The caller must select on it with a deadline.
func (n *CommitNotifier) Register(logIndex int) <-chan struct{} {
	ch := make(chan struct{})
	n.mu.Lock()
	n.chs[logIndex] = ch
	n.mu.Unlock()
	return ch
}

// Notify closes the channel registered for logIndex (if any), unblocking any
// goroutine waiting on it.  Safe to call even if no waiter is registered.
func (n *CommitNotifier) Notify(logIndex int) {
	n.mu.Lock()
	ch, ok := n.chs[logIndex]
	if ok {
		delete(n.chs, logIndex)
	}
	n.mu.Unlock()
	if ok {
		close(ch)
	}
}

// ── Applier ───────────────────────────────────────────────────────────────────

// Applier consumes committed log entries from the Raft applyCh and applies
// them to the local HNSWGraph.
//
// Every node in the cluster runs an Applier.  Because Raft guarantees that all
// nodes commit the same entries in the same order, every HNSW graph will
// converge to an identical state regardless of which node originally accepted
// the Insert RPC.
type Applier struct {
	applyCh  <-chan ApplyMsg
	graph    *vectorindex.HNSWGraph
	notifier *CommitNotifier
}

// NewApplier wires an Applier to the given Raft applyCh, HNSW graph, and
// notifier.  Call Run() in a separate goroutine.
func NewApplier(
	applyCh <-chan ApplyMsg,
	graph *vectorindex.HNSWGraph,
	notifier *CommitNotifier,
) *Applier {
	return &Applier{applyCh: applyCh, graph: graph, notifier: notifier}
}

// Run blocks forever, processing committed log entries.
// Call this as a goroutine: go applier.Run().
//
// For each committed entry the Applier:
//  1. Calls graph.Insert with the command's ID and vector.
//  2. Fires the CommitNotifier so the waiting gRPC handler can return success.
func (a *Applier) Run() {
	for msg := range a.applyCh {
		if !msg.CommandValid {
			// No-op entry (appended by a newly elected leader to establish
			// term authority).  Nothing to apply to the state machine.
			a.notifier.Notify(msg.CommandIndex)
			continue
		}

		cmd := msg.Command

		// Copy the vector slice so the graph owns it independently of this
		// ApplyMsg, which may be reused or GC'd after this function returns.
		vec := make([]float32, len(cmd.Vector))
		copy(vec, cmd.Vector)

		a.graph.Insert(cmd.ID, vec)

		log.Printf("[applier] applied index=%d id=%q dim=%d",
			msg.CommandIndex, cmd.ID, len(vec))

		// Signal any gRPC handler that was waiting on this log index.
		a.notifier.Notify(msg.CommandIndex)
	}
}
