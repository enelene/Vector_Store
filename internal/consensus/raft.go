// Package consensus implements the Raft distributed consensus algorithm.
//
// Reference:
//
//	Ongaro, D., & Ousterhout, J. (2014). "In Search of an Understandable
//	Consensus Algorithm (Extended Version)." USENIX ATC.
//	https://raft.github.io/raft.pdf
//
// This implementation covers §5 of the paper:
//   - Randomised leader election (§5.2)
//   - Log replication with fast conflict resolution (§5.3)
//   - Commit safety: only entries from the current term are committed (§5.4.2)
//
// Persistence (§5.5) and log compaction / snapshots (§7) are omitted;
// all state lives in memory and is lost on restart.
package consensus

import (
	"context"
	"encoding/json"
	"log"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"time"

	pb "vectordb/api/proto"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ── Timing constants ──────────────────────────────────────────────────────────
//
// Defaults are tuned for environments with non-trivial RTT jitter
// (e.g. kind/minikube on top of Docker Desktop, or any small VM). On a real
// LAN you can safely drop these to 150 ms / 300 ms / 50 ms / 100 ms — the
// algorithm is unchanged, the constants just need to be wider than the worst
// expected gRPC heartbeat round-trip.
const (
	// electionTimeoutMin / Max: randomised window for follower → candidate.
	// Wider than 2× heartbeat to guarantee at most one election per term.
	electionTimeoutMin = 400 * time.Millisecond
	electionTimeoutMax = 800 * time.Millisecond

	// heartbeatInterval: how often the leader broadcasts AppendEntries.
	// Must be significantly smaller than electionTimeoutMin.
	heartbeatInterval = 100 * time.Millisecond

	// rpcTimeout: deadline for a single Raft RPC call.
	rpcTimeout = 250 * time.Millisecond
)

// ── Node roles ────────────────────────────────────────────────────────────────

type role int32

const (
	follower  role = 0
	candidate role = 1
	leader    role = 2
)

// ── Log types ─────────────────────────────────────────────────────────────────

// InsertCommand is the only command type written to the Raft log.
// It is JSON-encoded before storage so the Raft layer stays unaware of
// higher-level types.
type InsertCommand struct {
	ID     string    `json:"id"`     // unique identifier of the vector to store
	Vector []float32 `json:"vector"` // the embedding payload
}

// logEntry is a single record in the replicated log.
// Command holds the raw JSON bytes of an InsertCommand, or nil for a no-op
// entry appended by a newly elected leader to commit prior-term entries.
type logEntry struct {
	Term    int
	Command []byte // JSON-encoded InsertCommand; nil = no-op
}

// ApplyMsg is sent on applyCh for every committed log entry.
type ApplyMsg struct {
	CommandValid bool          // false for no-ops
	Command      InsertCommand // valid when CommandValid == true
	CommandIndex int           // 0-based log index
	CommandTerm  int           // term in which the entry was originally appended
}

// ── Raft struct ───────────────────────────────────────────────────────────────

// Raft is a single node in a Raft cluster.
//
// Locking rule: rf.mu must be held when reading or writing any field except
// rf.dead (which uses atomic operations).
type Raft struct {
	pb.UnimplementedRaftServiceServer // satisfies the generated gRPC interface

	mu    sync.Mutex
	addrs []string               // peer addresses (index == node ID), including self
	peers []pb.RaftServiceClient // gRPC clients; peers[me] is nil (never used)
	me    int                    // this node's index in addrs

	// ── Persistent state (§5.5) — not yet persisted to disk ──────────────────
	currentTerm int
	votedFor    int // -1 = not voted this term
	log         []logEntry

	// ── Volatile state on all servers ────────────────────────────────────────
	commitIndex int // highest log index known to be committed (-1 = none)
	lastApplied int // highest log index applied to the state machine (-1 = none)

	// ── Volatile state on leaders (reinitialised on each election win) ────────
	nextIndex  []int // nextIndex[i]: next log index to send to peer i
	matchIndex []int // matchIndex[i]: highest log index known replicated on peer i

	role          role
	leaderId      int       // -1 = unknown
	heartbeatTime time.Time // when we last heard a valid leader heartbeat

	applyCh   chan ApplyMsg
	applyCond *sync.Cond // broadcasts when commitIndex advances

	dead int32 // atomic; 1 = killed

	// persistPath is the file Raft state is snapshotted to on graceful shutdown
	// and restored from on startup.  Empty string disables persistence.
	persistPath string
}

// Make creates and starts a Raft node.
//
//   - addrs:     addresses of ALL nodes (length == cluster size), including self.
//   - me:        this node's index in addrs.
//   - applyCh:   committed log entries are sent here; the caller must drain it.
//   - statePath: file used to snapshot/restore Raft state across restarts;
//     pass "" to disable persistence (e.g. ephemeral local runs).
func Make(addrs []string, me int, applyCh chan ApplyMsg, statePath string) *Raft {
	n := len(addrs)
	rf := &Raft{
		addrs:         addrs,
		peers:         make([]pb.RaftServiceClient, n),
		me:            me,
		currentTerm:   0,
		votedFor:      -1,
		log:           []logEntry{},
		commitIndex:   -1,
		lastApplied:   -1,
		nextIndex:     make([]int, n),
		matchIndex:    make([]int, n),
		role:          follower,
		leaderId:      -1,
		heartbeatTime: time.Now(),
		applyCh:       applyCh,
		persistPath:   statePath,
	}
	rf.applyCond = sync.NewCond(&rf.mu)

	// Establish gRPC connections to all peers (non-blocking; the actual TCP
	// handshake happens on the first RPC call, so peers need not be up yet).
	for i, addr := range addrs {
		if i == me {
			continue
		}
		conn, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Printf("[raft %d] dial %s: %v (will retry on first RPC)", me, addr, err)
			continue
		}
		rf.peers[i] = pb.NewRaftServiceClient(conn)
	}

	// Initialise matchIndex to -1 (nothing replicated yet).
	for i := range rf.matchIndex {
		rf.matchIndex[i] = -1
	}

	// Restore any snapshot left by a previous graceful shutdown.  Done before
	// the goroutines start (so no lock is needed) and before serving RPCs, so
	// the node resumes at its last durable term and commitApplyLoop can replay
	// committed entries [0..commitIndex] back into the HNSW state machine.
	rf.loadState()

	go rf.electionTicker()  // follower / candidate timeout loop
	go rf.heartbeatLoop()   // leader heartbeat loop
	go rf.commitApplyLoop() // commits committed entries to applyCh

	return rf
}

// ── Public API ────────────────────────────────────────────────────────────────

// Start proposes a new command to the cluster.
// Returns (logIndex, term, isLeader).
// If isLeader is false the command was NOT appended and will not be committed.
func (rf *Raft) Start(cmd InsertCommand) (index int, term int, isLeader bool) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	if rf.role != leader {
		return -1, -1, false
	}

	cmdBytes, _ := json.Marshal(cmd)
	entry := logEntry{Term: rf.currentTerm, Command: cmdBytes}
	rf.log = append(rf.log, entry)
	index = len(rf.log) - 1
	term = rf.currentTerm

	// The leader trivially holds the entry on its own log: record that in
	// matchIndex and try to advance the commit index. In a single-node cluster
	// this commits immediately (the leader alone is the majority); in a
	// multi-node cluster it is a harmless no-op until followers acknowledge.
	rf.matchIndex[rf.me] = index
	rf.advanceCommitIndex()

	// Kick off replication immediately without waiting for the heartbeat tick.
	go rf.broadcastAppendEntries(rf.currentTerm)

	return index, term, true
}

// GetState returns the node's current term and whether it believes it is leader.
func (rf *Raft) GetState() (term int, isLeader bool) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	return rf.currentTerm, rf.role == leader
}

// GetLeaderAddress returns the gRPC address of the node this server believes
// is the current leader, or "" if leadership is unknown.
func (rf *Raft) GetLeaderAddress() string {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	if rf.leaderId < 0 || rf.leaderId >= len(rf.addrs) {
		return ""
	}
	return rf.addrs[rf.leaderId]
}

// Kill stops the Raft node permanently (hard stop, used in tests).
// For an orderly stop that steps down and snapshots state, use Shutdown.
func (rf *Raft) Kill()        { atomic.StoreInt32(&rf.dead, 1) }
func (rf *Raft) killed() bool { return atomic.LoadInt32(&rf.dead) == 1 }

// ── Persistence & graceful shutdown ──────────────────────────────────────────

// persistentState is the on-disk snapshot written on graceful shutdown.
//
// In canonical Raft (§5.5), currentTerm, votedFor and the log are flushed to
// stable storage BEFORE responding to every RPC.  This implementation instead
// snapshots once, during graceful shutdown (the SIGTERM Kubernetes sends before
// it deletes a pod).  That is simpler and is sufficient here because cluster
// replication already provides durability against hard crashes: a node that is
// SIGKILL-ed loses its local snapshot but re-syncs the entire log from the
// current leader when it restarts.  The snapshot's role is therefore to make
// graceful restarts fast and to preserve the (currentTerm, votedFor) pair that
// Raft needs to avoid double-voting within a term across a restart.
//
// commitIndex is volatile in canonical Raft; we persist it only as an
// optimisation, so a restarting node can immediately replay committed entries
// into its HNSW graph instead of waiting for the leader's next AppendEntries.
type persistentState struct {
	CurrentTerm int        `json:"current_term"`
	VotedFor    int        `json:"voted_for"`
	Log         []logEntry `json:"log"`
	CommitIndex int        `json:"commit_index"`
}

// Shutdown performs an orderly stop of the Raft node:
//
//  1. If this node is the leader it steps down to follower so it stops emitting
//     heartbeats; the surviving nodes then elect a new leader within one
//     election timeout (≈150–300 ms).  We step down here, AFTER the caller has
//     drained in-flight client RPCs, so writes this leader already accepted can
//     still reach commit before it relinquishes the role.
//  2. The current state is snapshotted to disk for a fast restart.
//  3. The node is marked dead, ending its background goroutines.
func (rf *Raft) Shutdown() {
	rf.mu.Lock()
	if rf.role == leader {
		log.Printf("[raft %d] graceful shutdown: stepping down as leader (term %d)",
			rf.me, rf.currentTerm)
		rf.role = follower
		rf.leaderId = -1
	}
	rf.persistLocked()
	rf.mu.Unlock()

	rf.Kill() // mark dead → background goroutines exit
}

// persistLocked atomically writes the current state to persistPath
// (write-to-temp then rename).  Must be called with rf.mu held.
// Failures are logged but never fatal: the cluster can still recover this
// node's data by re-replicating the log from the leader.
func (rf *Raft) persistLocked() {
	if rf.persistPath == "" {
		return // persistence disabled
	}

	st := persistentState{
		CurrentTerm: rf.currentTerm,
		VotedFor:    rf.votedFor,
		Log:         rf.log,
		CommitIndex: rf.commitIndex,
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		log.Printf("[raft %d] persist: marshal failed: %v", rf.me, err)
		return
	}

	// Write to a temp file first, then rename.  Rename is atomic, so a reader
	// (or a crash mid-write) never observes a half-written snapshot.
	tmp := rf.persistPath + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		log.Printf("[raft %d] persist: write failed: %v", rf.me, err)
		return
	}
	if err := os.Rename(tmp, rf.persistPath); err != nil {
		log.Printf("[raft %d] persist: rename failed: %v", rf.me, err)
		return
	}
	log.Printf("[raft %d] persisted state: term=%d votedFor=%d log=%d commit=%d",
		rf.me, rf.currentTerm, rf.votedFor, len(rf.log), rf.commitIndex)
}

// loadState restores a snapshot written by a previous graceful shutdown.
// Called once from Make before the goroutines start, so it needs no lock.
// A missing file means a fresh node; a corrupt file is logged and ignored
// (the node starts fresh and re-syncs from the leader).
func (rf *Raft) loadState() {
	if rf.persistPath == "" {
		return
	}
	data, err := os.ReadFile(rf.persistPath)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Printf("[raft %d] load: read failed: %v (starting fresh)", rf.me, err)
		}
		return
	}
	var st persistentState
	if err := json.Unmarshal(data, &st); err != nil {
		log.Printf("[raft %d] load: corrupt snapshot: %v (starting fresh)", rf.me, err)
		return
	}

	rf.currentTerm = st.CurrentTerm
	rf.votedFor = st.VotedFor
	if st.Log != nil {
		rf.log = st.Log
	}
	rf.commitIndex = st.CommitIndex
	// lastApplied intentionally stays at -1 so commitApplyLoop replays
	// [0..commitIndex] into the HNSW graph, rebuilding the state machine.
	log.Printf("[raft %d] restored state: term=%d votedFor=%d log=%d commit=%d",
		rf.me, rf.currentTerm, rf.votedFor, len(rf.log), rf.commitIndex)
}

// ── gRPC RPC handlers (implement pb.RaftServiceServer) ───────────────────────

// RequestVote handles incoming vote solicitations (§5.2).
func (rf *Raft) RequestVote(_ context.Context, args *pb.VoteRequest) (*pb.VoteResponse, error) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	resp := &pb.VoteResponse{Term: int32(rf.currentTerm)}

	// Rule 1: any RPC with a higher term causes us to revert to follower.
	if int(args.Term) > rf.currentTerm {
		rf.stepDown(int(args.Term))
	}

	// Reject stale candidates immediately.
	if int(args.Term) < rf.currentTerm {
		return resp, nil
	}

	// Already voted for someone else this term.
	if rf.votedFor != -1 && rf.votedFor != int(args.CandidateId) {
		return resp, nil
	}

	// Candidate's log must be at least as up-to-date as ours (§5.4.1).
	myLastIdx, myLastTerm := rf.lastLogIndexTerm()
	candLastTerm := int(args.LastLogTerm)
	candLastIdx := int(args.LastLogIndex)
	upToDate := candLastTerm > myLastTerm ||
		(candLastTerm == myLastTerm && candLastIdx >= myLastIdx)
	if !upToDate {
		return resp, nil
	}

	// Grant vote.
	rf.votedFor = int(args.CandidateId)
	rf.heartbeatTime = time.Now() // reset election timer on granting vote
	resp.VoteGranted = true
	resp.Term = int32(rf.currentTerm)
	return resp, nil
}

// AppendEntries handles heartbeats and log replication from the leader (§5.3).
func (rf *Raft) AppendEntries(_ context.Context, args *pb.AppendRequest) (*pb.AppendResponse, error) {
	rf.mu.Lock()
	defer rf.mu.Unlock()

	resp := &pb.AppendResponse{
		Term:         int32(rf.currentTerm),
		ConflictTerm: -1,
	}

	// Reject stale leaders.
	if int(args.Term) < rf.currentTerm {
		return resp, nil
	}

	// Recognise valid leader and reset election timer.
	if int(args.Term) >= rf.currentTerm {
		rf.stepDown(int(args.Term))
		rf.leaderId = int(args.LeaderId)
	}
	rf.heartbeatTime = time.Now()
	resp.Term = int32(rf.currentTerm)

	// Log consistency check (§5.3).
	prevIdx := int(args.PrevLogIndex)
	prevTerm := int(args.PrevLogTerm)

	if prevIdx >= 0 {
		if prevIdx >= len(rf.log) {
			// Our log is shorter than leader expects.
			resp.ConflictIndex = int32(len(rf.log))
			resp.ConflictTerm = -1
			return resp, nil
		}
		if rf.log[prevIdx].Term != prevTerm {
			// Conflict: return the first index of the conflicting term so the
			// leader can skip the entire term in one round-trip.
			ct := rf.log[prevIdx].Term
			ci := prevIdx
			for ci > 0 && rf.log[ci-1].Term == ct {
				ci--
			}
			resp.ConflictTerm = int32(ct)
			resp.ConflictIndex = int32(ci)
			return resp, nil
		}
	}

	// Append or overwrite entries (§5.3 — delete conflicting suffix first).
	for i, protoEntry := range args.Entries {
		idx := prevIdx + 1 + i
		e := logEntry{Term: int(protoEntry.Term), Command: protoEntry.Command}
		if idx < len(rf.log) {
			if rf.log[idx].Term != e.Term {
				rf.log = append(rf.log[:idx], e) // truncate then append
			}
			// Identical entry already present — skip.
		} else {
			rf.log = append(rf.log, e)
		}
	}

	// Advance commitIndex using the leader's leaderCommit (§5.3).
	lc := int(args.LeaderCommit)
	if lc > rf.commitIndex {
		newCommit := lc
		if lastIdx := len(rf.log) - 1; lastIdx < newCommit {
			newCommit = lastIdx
		}
		rf.commitIndex = newCommit
		rf.applyCond.Signal()
	}

	resp.Success = true
	return resp, nil
}

// ── Internal goroutines ───────────────────────────────────────────────────────

// electionTicker periodically checks whether the election timeout has elapsed.
// When it has (and we are not leader) we start a new election (§5.2).
func (rf *Raft) electionTicker() {
	for !rf.killed() {
		timeout := electionTimeoutMin +
			time.Duration(rand.Int63n(int64(electionTimeoutMax-electionTimeoutMin)))
		start := time.Now()
		time.Sleep(timeout)

		rf.mu.Lock()
		if rf.role != leader && rf.heartbeatTime.Before(start) {
			// No valid heartbeat received during the sleep window → start election.
			rf.startElection()
		}
		rf.mu.Unlock()
	}
}

// heartbeatLoop sends AppendEntries to all peers every heartbeatInterval when
// this node is the leader.  Empty entries act as heartbeats; non-empty entries
// replicate new log commands.
func (rf *Raft) heartbeatLoop() {
	for !rf.killed() {
		rf.mu.Lock()
		if rf.role == leader {
			term := rf.currentTerm
			rf.mu.Unlock()
			go rf.broadcastAppendEntries(term)
		} else {
			rf.mu.Unlock()
		}
		time.Sleep(heartbeatInterval)
	}
}

// commitApplyLoop waits for commitIndex to advance past lastApplied, then
// sends the newly committed entries to applyCh for the state machine to apply.
func (rf *Raft) commitApplyLoop() {
	for !rf.killed() {
		rf.mu.Lock()
		for rf.commitIndex <= rf.lastApplied {
			rf.applyCond.Wait()
		}
		// Collect all pending entries under the lock.
		start := rf.lastApplied + 1
		end := rf.commitIndex
		batch := make([]logEntry, end-start+1)
		copy(batch, rf.log[start:end+1])
		rf.mu.Unlock()

		// Send each entry to applyCh without holding the mutex so the state
		// machine can call rf.Start() freely.
		for i, e := range batch {
			msg := ApplyMsg{CommandIndex: start + i, CommandTerm: e.Term}
			if len(e.Command) > 0 {
				if err := json.Unmarshal(e.Command, &msg.Command); err == nil {
					msg.CommandValid = true
				}
			}
			rf.applyCh <- msg
		}

		rf.mu.Lock()
		if end > rf.lastApplied {
			rf.lastApplied = end
		}
		rf.mu.Unlock()
	}
}

// ── Election ──────────────────────────────────────────────────────────────────

// startElection transitions this node to candidate, increments the term, and
// sends RequestVote RPCs to all peers.  Called with rf.mu held.
func (rf *Raft) startElection() {
	rf.currentTerm++
	rf.role = candidate
	rf.votedFor = rf.me
	rf.leaderId = -1
	rf.heartbeatTime = time.Now() // prevent re-triggering during this election

	term := rf.currentTerm
	lastIdx, lastTerm := rf.lastLogIndexTerm()
	votes := 1 // voted for self
	majority := len(rf.addrs)/2 + 1

	log.Printf("[raft %d] starting election for term %d", rf.me, term)

	for i := range rf.addrs {
		if i == rf.me || rf.peers[i] == nil {
			continue
		}
		go rf.sendRequestVote(i, term, lastIdx, lastTerm, &votes, majority)
	}

	// If our own vote already constitutes a majority — e.g. a single-node
	// cluster — no RequestVote RPCs are sent, so win the election here rather
	// than waiting for a peer reply that will never arrive. Safe to read votes
	// without synchronisation: we still hold rf.mu, so the goroutines spawned
	// above cannot have incremented it yet.
	if votes >= majority {
		rf.becomeLeader()
	}
}

// sendRequestVote sends a single RequestVote RPC and processes the reply.
// Runs in its own goroutine (must not hold rf.mu when called).
func (rf *Raft) sendRequestVote(peer, term, lastIdx, lastTerm int, votes *int, majority int) {
	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()

	resp, err := rf.peers[peer].RequestVote(ctx, &pb.VoteRequest{
		Term:         int32(term),
		CandidateId:  int32(rf.me),
		LastLogIndex: int32(lastIdx),
		LastLogTerm:  int32(lastTerm),
	})
	if err != nil {
		return
	}

	rf.mu.Lock()
	defer rf.mu.Unlock()

	if int(resp.Term) > rf.currentTerm {
		rf.stepDown(int(resp.Term))
		return
	}
	// Discard stale replies (election term changed or we're no longer candidate).
	if rf.role != candidate || rf.currentTerm != term {
		return
	}
	if !resp.VoteGranted {
		return
	}

	*votes++
	if *votes >= majority {
		rf.becomeLeader()
	}
}

// ── Leader helpers ────────────────────────────────────────────────────────────

// becomeLeader transitions this node to leader and initialises per-peer state.
// Called with rf.mu held.
func (rf *Raft) becomeLeader() {
	if rf.role != candidate {
		return // already became leader or stepped down
	}
	rf.role = leader
	rf.leaderId = rf.me
	log.Printf("[raft %d] became leader for term %d", rf.me, rf.currentTerm)

	// nextIndex: next entry to send to each peer (start optimistically at end of log).
	// matchIndex: highest entry known to be replicated on each peer.
	for i := range rf.addrs {
		rf.nextIndex[i] = len(rf.log)
		rf.matchIndex[i] = -1
	}
	// Leader's own log is always fully replicated.
	if len(rf.log) > 0 {
		rf.matchIndex[rf.me] = len(rf.log) - 1
	}

	// Append a no-op entry to quickly commit prior-term entries (§5.4.2).
	rf.log = append(rf.log, logEntry{Term: rf.currentTerm, Command: nil})
	rf.matchIndex[rf.me] = len(rf.log) - 1

	// Single-node clusters have no peers to ack, so commit the no-op (and any
	// carried-over entries) now; multi-node clusters wait for follower acks.
	rf.advanceCommitIndex()
}

// broadcastAppendEntries spawns one goroutine per peer to send AppendEntries.
// term is captured at the moment of the call to detect stale goroutines.
func (rf *Raft) broadcastAppendEntries(term int) {
	for i := range rf.addrs {
		if i == rf.me || rf.peers[i] == nil {
			continue
		}
		go rf.replicateToPeer(i, term)
	}
}

// replicateToPeer sends one round of AppendEntries to a single peer.
// Handles both heartbeats (empty entries) and actual log replication.
func (rf *Raft) replicateToPeer(peer, term int) {
	rf.mu.Lock()

	// Abort if we are no longer the leader for this term.
	if rf.role != leader || rf.currentTerm != term {
		rf.mu.Unlock()
		return
	}

	nextIdx := rf.nextIndex[peer]
	prevLogIdx := nextIdx - 1
	prevLogTerm := 0
	if prevLogIdx >= 0 && prevLogIdx < len(rf.log) {
		prevLogTerm = rf.log[prevLogIdx].Term
	}

	// Snapshot the entries to send (may be empty for a pure heartbeat).
	var entriesToSend []logEntry
	if nextIdx < len(rf.log) {
		entriesToSend = make([]logEntry, len(rf.log)-nextIdx)
		copy(entriesToSend, rf.log[nextIdx:])
	}
	leaderCommit := rf.commitIndex
	rf.mu.Unlock()

	// Convert to proto.
	protoEntries := make([]*pb.RaftLogEntry, len(entriesToSend))
	for i, e := range entriesToSend {
		protoEntries[i] = &pb.RaftLogEntry{Term: int32(e.Term), Command: e.Command}
	}

	ctx, cancel := context.WithTimeout(context.Background(), rpcTimeout)
	defer cancel()

	resp, err := rf.peers[peer].AppendEntries(ctx, &pb.AppendRequest{
		Term:         int32(term),
		LeaderId:     int32(rf.me),
		PrevLogIndex: int32(prevLogIdx),
		PrevLogTerm:  int32(prevLogTerm),
		Entries:      protoEntries,
		LeaderCommit: int32(leaderCommit),
	})
	if err != nil {
		return
	}

	rf.mu.Lock()
	defer rf.mu.Unlock()

	if int(resp.Term) > rf.currentTerm {
		rf.stepDown(int(resp.Term))
		return
	}
	if rf.role != leader || rf.currentTerm != term {
		return // stale response
	}

	if resp.Success {
		newMatch := prevLogIdx + len(entriesToSend)
		if newMatch > rf.matchIndex[peer] {
			rf.matchIndex[peer] = newMatch
		}
		rf.nextIndex[peer] = rf.matchIndex[peer] + 1
		rf.advanceCommitIndex()
	} else {
		// Fast backtrack: jump to the first index of the conflicting term, or
		// to the length of the follower's log if it is too short.
		ct := int(resp.ConflictTerm)
		ci := int(resp.ConflictIndex)
		if ct < 0 {
			// Follower log too short.
			rf.nextIndex[peer] = ci
		} else {
			// Find last entry in our log with conflictTerm; skip past it.
			lastInTerm := -1
			for j := len(rf.log) - 1; j >= 0; j-- {
				if rf.log[j].Term == ct {
					lastInTerm = j
					break
				}
			}
			if lastInTerm >= 0 {
				rf.nextIndex[peer] = lastInTerm + 1
			} else {
				rf.nextIndex[peer] = ci
			}
		}
		// nextIndex == 0 means "send the whole log from index 0", which is the
		// correct response when the follower's log is empty (its conflictIndex is
		// then 0). Only NEGATIVE values are invalid, so clamp at zero, not one.
		if rf.nextIndex[peer] < 0 {
			rf.nextIndex[peer] = 0
		}
	}
}

// advanceCommitIndex checks whether any new log index can be committed —
// i.e., a majority of nodes have replicated the entry AND the entry was
// appended in the current term (§5.4.2 safety rule).
// Called with rf.mu held.
func (rf *Raft) advanceCommitIndex() {
	majority := len(rf.addrs)/2 + 1
	// Scan backwards from end of log to find the highest committable index.
	for n := len(rf.log) - 1; n > rf.commitIndex; n-- {
		if rf.log[n].Term != rf.currentTerm {
			continue // §5.4.2: never commit by counting replicas from prior terms
		}
		count := 0
		for i := range rf.addrs {
			if rf.matchIndex[i] >= n {
				count++
			}
		}
		if count >= majority {
			rf.commitIndex = n
			rf.applyCond.Signal()
			break // advancing past the first found is done in the next call
		}
	}
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// stepDown reverts this node to follower state at the given term.
// Called with rf.mu held.
func (rf *Raft) stepDown(newTerm int) {
	rf.currentTerm = newTerm
	rf.role = follower
	rf.votedFor = -1
	rf.leaderId = -1
}

// lastLogIndexTerm returns the index and term of the last log entry,
// or (-1, 0) if the log is empty. Called with rf.mu held.
func (rf *Raft) lastLogIndexTerm() (index, term int) {
	if len(rf.log) == 0 {
		return -1, 0
	}
	last := len(rf.log) - 1
	return last, rf.log[last].Term
}
