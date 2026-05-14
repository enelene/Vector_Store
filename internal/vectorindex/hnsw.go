// Package vectorindex implements the core search engine.
//
// This file contains the Hierarchical Navigable Small World (HNSW) graph, a
// proximity graph algorithm for approximate nearest-neighbour (ANN) search.
//
// Reference:
//
//	Malkov, Y. A., & Yashunin, D. A. (2018).
//	"Efficient and robust approximate nearest neighbor search using
//	 Hierarchical Navigable Small World graphs."
//	IEEE TPAMI. arXiv:1603.09320
package vectorindex

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
	"sync"
)

// ──────────────────────────────────────────────────────────────────────────────
// Algorithm parameters
// ──────────────────────────────────────────────────────────────────────────────

// Default HNSW hyper-parameters from the original paper.
// M=16 / efConstruction=200 are the production sweet-spot for most embedding
// workloads (high recall with manageable memory and build time).
const (
	DefaultM              = 16  // max bidirectional edges per node per layer
	DefaultEfConstruction = 200 // candidate pool size during graph construction
	DefaultEfSearch       = 50  // candidate pool size during queries
)

// ──────────────────────────────────────────────────────────────────────────────
// Priority-queue internals
// ──────────────────────────────────────────────────────────────────────────────

// candidate pairs a stored vector ID with its cosine similarity to a query.
// Higher score ≡ closer in the navigable-small-world sense.
type candidate struct {
	id    string
	score float32
}

// explorerHeap is a max-heap ordered by score.
//
// Popping gives the BEST (highest-similarity) candidate, so we always expand
// the most-promising node first — equivalent to a greedy best-first traversal.
type explorerHeap []candidate

func (h explorerHeap) Len() int           { return len(h) }
func (h explorerHeap) Less(i, j int) bool { return h[i].score > h[j].score } // max-heap
func (h explorerHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *explorerHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *explorerHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// resultsHeap is a min-heap ordered by score.
//
// The WORST result (lowest similarity) sits at index 0 at all times, making
// it O(log ef) to evict when the heap exceeds its ef capacity limit.
type resultsHeap []candidate

func (h resultsHeap) Len() int           { return len(h) }
func (h resultsHeap) Less(i, j int) bool { return h[i].score < h[j].score } // min-heap
func (h resultsHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultsHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *resultsHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// ──────────────────────────────────────────────────────────────────────────────
// Node
// ──────────────────────────────────────────────────────────────────────────────

// Node stores a single vector and its adjacency lists at every layer it
// participates in.
//
// Immutability contract:
//
//	id       — set once at creation, never changes.
//	vector   — set once at creation, never changes.
//	            Safe to read from any goroutine WITHOUT holding mu.
//	maxLayer — set once at creation, never changes.
//
//	neighbors — mutable; MUST be accessed under mu.
type Node struct {
	id       string
	vector   []float32 // immutable — readable without mu
	maxLayer int       // immutable

	// neighbors[layer] is the ordered list of neighbour IDs at that layer.
	// Indexed 0 … maxLayer inclusive.  Protected by mu.
	neighbors [][]string
	mu        sync.RWMutex
}

// ──────────────────────────────────────────────────────────────────────────────
// HNSWGraph
// ──────────────────────────────────────────────────────────────────────────────

// HNSWGraph is a Hierarchical Navigable Small World graph.
//
// ┌─────────────────────────── LOCK ORDERING RULE ───────────────────────────┐
// │                                                                           │
// │   g.mu  MUST be acquired BEFORE any node.mu.                             │
// │   No code path may acquire g.mu while already holding a node.mu.         │
// │                                                                           │
// │   This turns the locking graph into a DAG (g.mu → node.mu), which        │
// │   is a sufficient condition for deadlock freedom:                         │
// │   circular wait requires a cycle, and a DAG has none.                    │
// │                                                                           │
// └───────────────────────────────────────────────────────────────────────────┘
//
// Consequence for readers of this code: whenever you see node.mu acquired
// inside a g.mu.RLock section, that is intentional and safe.  The reverse
// (g.mu inside node.mu) is always a bug.
type HNSWGraph struct {
	mu         sync.RWMutex
	nodes      map[string]*Node // protected by mu
	entryPoint string           // ID of the graph entry point; protected by mu
	maxLayer   int              // highest layer in the graph; protected by mu

	// Algorithm hyper-parameters — immutable after construction.
	M              int     // max edges per node on layers 1+
	Mmax0          int     // max edges per node on layer 0 (= 2×M)
	efConstruction int     // candidate pool size used during Insert
	efSearch       int     // candidate pool size used during Search
	mL             float64 // level normalisation factor = 1/ln(M)

	rng   *rand.Rand // not safe for concurrent use; protected by rngMu
	rngMu sync.Mutex
}

// SearchResult is the public result type returned by Search.
type SearchResult struct {
	ID    string  // identifier of the matched vector
	Score float32 // cosine similarity to the query, in [-1, 1]; higher is closer
}

// NewHNSWGraph creates an empty HNSW graph with the given parameters.
//
//   - M              max edge degree per node (use DefaultM = 16 if unsure).
//   - efConstruction candidate pool during inserts (use DefaultEfConstruction = 200).
//
// Higher M → better recall, more memory.
// Higher efConstruction → better graph quality, slower inserts.
func NewHNSWGraph(M, efConstruction int) *HNSWGraph {
	if M <= 0 {
		M = DefaultM
	}
	if efConstruction <= 0 {
		efConstruction = DefaultEfConstruction
	}
	return &HNSWGraph{
		nodes:          make(map[string]*Node),
		M:              M,
		Mmax0:          2 * M,
		efConstruction: efConstruction,
		efSearch:       DefaultEfSearch,
		mL:             1.0 / math.Log(float64(M)),
		rng:            rand.New(rand.NewSource(42)),
	}
}

// maxLinksAtLayer returns the maximum allowed edge degree at a given layer.
// Layer 0 (the base graph) allows 2×M edges so that the densest proximity
// graph has sufficient connectivity for high-recall neighbourhood searches.
func (g *HNSWGraph) maxLinksAtLayer(layer int) int {
	if layer == 0 {
		return g.Mmax0
	}
	return g.M
}

// ──────────────────────────────────────────────────────────────────────────────
// Level assignment
// ──────────────────────────────────────────────────────────────────────────────

// assignLevel draws an insertion level from the exponential distribution.
//
// Formula (§4.1 of the HNSW paper):
//
//	l = ⌊ −ln(uniform(0,1)) × mL ⌋,   mL = 1/ln(M)
//
// This gives P(l ≥ k) = (1/M)^k, so the expected node count at layer k
// decreases by a factor of M for each step up — an exponential hierarchy.
// Most nodes land at layer 0; very few reach high layers, keeping traversal
// logarithmic in the total node count.
func (g *HNSWGraph) assignLevel() int {
	g.rngMu.Lock()
	r := g.rng.Float64()
	g.rngMu.Unlock()

	// Guard against ln(0) = -∞.
	if r < 1e-10 {
		r = 1e-10
	}
	return int(-math.Log(r) * g.mL)
}

// ──────────────────────────────────────────────────────────────────────────────
// Insert   (Algorithm 1, Malkov & Yashunin 2018)
// ──────────────────────────────────────────────────────────────────────────────

// Insert adds the vector to the graph, connecting it to its nearest neighbours
// at every layer up to the randomly assigned level.
//
// Concurrency design — four phases with explicit lock transitions:
//
//	Phase 1 │ g.mu.Lock()   Add the node to the map and bump maxLayer.
//	        │               The new node is reachable but has empty
//	        │               neighbour lists — searches see it but cannot
//	        │               route through it yet.  entryPoint is NOT
//	        │               updated here; the node must be fully wired first.
//	        │ g.mu.Unlock()
//
//	Phase 2 │ (no graph lock) Greedy single-NN descent from the current entry
//	        │ point through layers above the new node's level.  Each step
//	        │ calls searchLayer, which takes its own RLock internally.
//	        │ This identifies the best entry point for the connection phase.
//
//	Phase 3 │ Per-layer: searchLayer (RLock inside) → connect bidirectional
//	        │ edges → prune over-full neighbour lists.
//	        │
//	        │ Back-connections are added via addBackConnection, which follows
//	        │ the lock ordering rule: g.mu.RLock() first, then node.mu.Lock().
//
//	Phase 4 │ g.mu.Lock()   Promote new node to entry point iff its layer
//	        │               exceeds the current maximum.  Re-checked under
//	        │               the lock to handle concurrent high-layer inserts.
//	        │ g.mu.Unlock()
func (g *HNSWGraph) Insert(id string, vector []float32) {
	level := g.assignLevel()

	// Build the node with pre-allocated (empty) neighbour slices for each layer.
	newNode := &Node{
		id:        id,
		vector:    vector,
		maxLayer:  level,
		neighbors: make([][]string, level+1),
	}
	for i := range newNode.neighbors {
		newNode.neighbors[i] = make([]string, 0, g.maxLinksAtLayer(i))
	}

	// ── Phase 1: register node in the graph ──────────────────────────────────
	g.mu.Lock()

	if len(g.nodes) == 0 {
		// First insertion: the node trivially becomes the sole entry point.
		g.nodes[id] = newNode
		g.entryPoint = id
		g.maxLayer = level
		g.mu.Unlock()
		return
	}

	// Capture a consistent snapshot of the current topology entry state.
	ep := g.entryPoint
	curMaxLayer := g.maxLayer

	g.nodes[id] = newNode

	// Bump maxLayer eagerly so that concurrent searches enter at the correct
	// layer even before the entry point pointer is updated.  The entry point
	// itself is updated only in Phase 4, after connections are fully wired.
	if level > curMaxLayer {
		g.maxLayer = level
	}

	g.mu.Unlock()

	// ── Phase 2: greedy descent through layers above newNode's level ──────────
	// Use ef=1 (pure greedy) to cheaply find a good entry point for Phase 3.
	// We only need the single nearest neighbour at each upper layer; there is
	// no point in a wide search because we will not connect at these layers.
	for lc := curMaxLayer; lc > level; lc-- {
		upper := g.searchLayer(ep, vector, 1, lc)
		if len(upper) > 0 {
			ep = upper[0].id
		}
	}

	// ── Phase 3: connect at every layer from level down to 0 ─────────────────
	for lc := min(level, curMaxLayer); lc >= 0; lc-- {
		// Find the efConstruction nearest neighbours at this layer.
		candidates := g.searchLayer(ep, vector, g.efConstruction, lc)
		if len(candidates) == 0 {
			continue
		}

		// Simple neighbour selection: keep the best M (or Mmax0) candidates.
		// The slice returned by searchLayer is already sorted descending.
		neighbours := selectNeighbors(candidates, g.maxLinksAtLayer(lc))

		// Write the new node's outgoing edges at this layer.
		//
		// Safety: newNode is in the graph map (Phase 1), but no existing node
		// has a back-edge to newNode yet — those are wired immediately below.
		// Therefore no concurrent goroutine can be waiting on newNode.mu for
		// a back-connection write at this layer, making this write safe without
		// holding g.mu.  (Lock ordering: we hold no lock → acquire node lock.)
		ids := make([]string, len(neighbours))
		for i, c := range neighbours {
			ids[i] = c.id
		}
		newNode.mu.Lock()
		newNode.neighbors[lc] = ids
		newNode.mu.Unlock()

		// Wire back-connections: each selected neighbour must also point to
		// newNode, turning the graph bidirectional.
		//
		// addBackConnection acquires g.mu.RLock → nbr.mu in that order.
		// We call it once per neighbour, releasing all locks between calls,
		// so we never hold more than one node lock at a time.
		// This one-at-a-time discipline eliminates any possibility of the
		// classic "hold-and-wait" deadlock pattern across node locks.
		for i := range neighbours {
			g.addBackConnection(neighbours[i].id, id, vector, lc)
		}

		// The best candidate at this layer becomes the entry point for the
		// next (lower) layer — this is the greedy descent step of Algorithm 1.
		ep = candidates[0].id
	}

	// ── Phase 4: promote newNode to entry point if it has the highest layer ───
	// We defer this until all connections are in place so that any search
	// entering via the entry point immediately finds a well-connected node.
	if level > curMaxLayer {
		g.mu.Lock()
		// Re-check: a concurrent insert may have set an even higher entry point.
		if level >= g.maxLayer {
			g.entryPoint = id
		}
		g.mu.Unlock()
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// addBackConnection — bidirectional edge wiring with safe prune
// ──────────────────────────────────────────────────────────────────────────────

// addBackConnection appends newID to nbrID's neighbour list at layer lc,
// then prunes the list down to maxLinksAtLayer if it has grown too large.
//
// Lock protocol (two phases, strictly respecting g.mu → node.mu ordering):
//
//	Phase A │ g.mu.RLock()
//	        │   — look up nbrID; bail if missing or layer out of range.
//	        │   — nbr.mu.RLock() → snapshot current neighbour list.
//	        │   — nbr.mu.RUnlock()
//	        │   — score all candidates (new node + existing neighbours) using
//	        │     nbr.vector, which is immutable.
//	        │ g.mu.RUnlock()
//	        │
//	Phase B │ nbr.mu.Lock()    ← no g.mu held here
//	        │   — append newID.
//	        │   — prune with pre-scored candidates if over the degree limit.
//	        │ nbr.mu.Unlock()
//
// Why this cannot deadlock:
//
//	The only simultaneous lock pair taken anywhere in the code is
//	(g.mu.RLock, node.mu.RLock), always in that order.  Phase B holds ONLY
//	node.mu.Lock with no graph lock, so a goroutine in Phase B can never
//	block a goroutine waiting for g.mu.Lock — the classic deadlock cycle
//	requires both goroutines to hold one lock each and want the other's.
//	Since no goroutine ever holds node.mu while wanting g.mu, no such cycle
//	can form.
//
// Approximation note:
//
//	Between Phase A (snapshot) and Phase B (write), another goroutine may
//	have added or pruned edges on the same neighbour.  The pre-scored
//	candidate list may therefore be slightly stale when pruning is applied.
//	This is an accepted property of concurrent HNSW: graph quality under
//	high contention is slightly lower than sequential construction, but the
//	structure remains valid and navigable.
func (g *HNSWGraph) addBackConnection(nbrID, newID string, newVec []float32, lc int) {
	// ── Phase A: read-only data collection ───────────────────────────────────
	g.mu.RLock()

	nbr, ok := g.nodes[nbrID]
	if !ok || lc > nbr.maxLayer {
		// Neighbour does not exist at this layer — nothing to wire.
		g.mu.RUnlock()
		return
	}

	// nbr.vector is immutable: safe to read without nbr.mu.
	newScore := CosineSimilarity(nbr.vector, newVec)
	maxLinks := g.maxLinksAtLayer(lc)

	// Snapshot the neighbour's current edge list.
	// nbr.mu.RLock is acquired while g.mu.RLock is held — this follows the
	// g.mu → node.mu ordering rule and is therefore safe.
	nbr.mu.RLock()
	existingIDs := make([]string, len(nbr.neighbors[lc]))
	copy(existingIDs, nbr.neighbors[lc])
	nbr.mu.RUnlock()

	// Pre-score the full candidate set: the new node plus every existing edge.
	// We do this under the graph read lock so that each neighbour's (immutable)
	// vector can be looked up safely.
	allCandidates := make([]candidate, 0, len(existingIDs)+1)
	allCandidates = append(allCandidates, candidate{id: newID, score: newScore})
	for _, eid := range existingIDs {
		if en, exists := g.nodes[eid]; exists {
			allCandidates = append(allCandidates, candidate{
				id:    eid,
				score: CosineSimilarity(nbr.vector, en.vector),
			})
		}
	}

	g.mu.RUnlock() // Release graph lock BEFORE acquiring node write lock.

	// ── Phase B: mutate the neighbour node ───────────────────────────────────
	nbr.mu.Lock()
	defer nbr.mu.Unlock()

	// Append the new back-edge.
	nbr.neighbors[lc] = append(nbr.neighbors[lc], newID)

	// Early exit: no pruning needed.
	if len(nbr.neighbors[lc]) <= maxLinks {
		return
	}

	// Prune: keep the best maxLinks neighbours by cosine similarity.
	// Sort descending so the strongest connections survive.
	sort.Slice(allCandidates, func(i, j int) bool {
		return allCandidates[i].score > allCandidates[j].score
	})
	if len(allCandidates) > maxLinks {
		allCandidates = allCandidates[:maxLinks]
	}
	pruned := make([]string, len(allCandidates))
	for i, c := range allCandidates {
		pruned[i] = c.id
	}
	nbr.neighbors[lc] = pruned
}

// ──────────────────────────────────────────────────────────────────────────────
// Search   (Algorithm 5, Malkov & Yashunin 2018)
// ──────────────────────────────────────────────────────────────────────────────

// Search returns the topK nearest neighbours of query using greedy layer
// routing.  The algorithm descends from the top layer to layer 1 using a
// single-NN greedy walk, then performs a full ef-beam search at layer 0.
//
// The ef parameter used at layer 0 is max(topK, g.efSearch), ensuring that the
// result set is never artificially constrained below topK.
func (g *HNSWGraph) Search(query []float32, topK int) []SearchResult {
	// Snapshot entry state under a brief read lock.
	g.mu.RLock()
	ep := g.entryPoint
	curMaxLayer := g.maxLayer
	g.mu.RUnlock()

	if ep == "" {
		return nil // graph is empty
	}

	ef := g.efSearch
	if topK > ef {
		// Widen the beam if the caller wants more results than the default ef.
		ef = topK
	}

	// ── Upper layers: greedy single-NN descent (ef=1) ─────────────────────────
	// We are not yet at the target layer; use ef=1 to quickly converge toward
	// the neighbourhood of the query without expensive wide searches.
	for lc := curMaxLayer; lc > 0; lc-- {
		upper := g.searchLayer(ep, query, 1, lc)
		if len(upper) > 0 {
			ep = upper[0].id
		}
	}

	// ── Layer 0: wide beam search with full ef ────────────────────────────────
	// Layer 0 contains every node and has the densest connections (Mmax0 = 2M
	// edges), so this is where the accurate nearest-neighbour set is found.
	candidates := g.searchLayer(ep, query, ef, 0)

	// Truncate to the requested topK (candidates already sorted descending).
	if len(candidates) > topK {
		candidates = candidates[:topK]
	}

	out := make([]SearchResult, len(candidates))
	for i, c := range candidates {
		out[i] = SearchResult{ID: c.id, Score: c.score}
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// searchLayer   (Algorithm 2, Malkov & Yashunin 2018)
// ──────────────────────────────────────────────────────────────────────────────

// searchLayer executes a greedy beam search starting at epID on the given
// layer and returns at most ef results sorted by descending cosine similarity.
//
// Data structure recap:
//   - exp  (explorerHeap, max-heap)  — candidates we still need to expand;
//     we always expand the BEST (highest-score) candidate next.
//   - res  (resultsHeap, min-heap)   — best ef results found so far;
//     the root is the WORST result, enabling O(log ef) eviction.
//
// Termination: once the best unexplored candidate is worse than the worst
// result AND the result set is full, no future expansion can improve the
// result set — we stop.
//
// Thread safety:
//   - g.mu.RLock is held for the ENTIRE call, giving a consistent view of
//     the node map.  (Insertions that complete after the lock is taken are
//     simply not visible to this search, which is correct ANN behaviour.)
//   - node.mu.RLock is acquired/released for each neighbour-list read,
//     following the g.mu → node.mu ordering rule.
//   - No write locks are ever taken inside this function.
func (g *HNSWGraph) searchLayer(epID string, query []float32, ef, layer int) []candidate {
	g.mu.RLock()
	defer g.mu.RUnlock()

	epNode, ok := g.nodes[epID]
	if !ok {
		return nil
	}

	// epNode.vector is immutable — no lock needed.
	epScore := CosineSimilarity(query, epNode.vector)

	// visited prevents re-expansion of nodes we have already scored.
	visited := make(map[string]struct{}, ef*2)
	visited[epID] = struct{}{}

	// Initialise both heaps with the entry point.
	exp := &explorerHeap{candidate{id: epID, score: epScore}}
	heap.Init(exp)

	res := &resultsHeap{candidate{id: epID, score: epScore}}
	heap.Init(res)

	for exp.Len() > 0 {
		cur := heap.Pop(exp).(candidate) // best unexplored candidate

		// Termination condition (Algorithm 2, line 9):
		// res[0] is the WORST result in our min-heap.  If the best unexplored
		// candidate (cur) is already worse than that, expanding further can
		// only produce results worse than what we have — stop.
		if res.Len() >= ef && cur.score < (*res)[0].score {
			break
		}

		// Expand: inspect all neighbours of cur at this layer.
		curNode, exists := g.nodes[cur.id]
		if !exists {
			continue
		}

		// Read the neighbour list under the node read lock (g.mu already held).
		curNode.mu.RLock()
		var nbrIDs []string
		if layer < len(curNode.neighbors) {
			nbrIDs = make([]string, len(curNode.neighbors[layer]))
			copy(nbrIDs, curNode.neighbors[layer])
		}
		curNode.mu.RUnlock()

		for _, nbrID := range nbrIDs {
			if _, seen := visited[nbrID]; seen {
				continue
			}
			visited[nbrID] = struct{}{}

			nbrNode, exists := g.nodes[nbrID]
			if !exists {
				continue
			}
			// nbrNode.vector is immutable — safe without lock.
			score := CosineSimilarity(query, nbrNode.vector)

			// Admit this neighbour if the result set is not full yet, or if it
			// beats the current worst result.
			if res.Len() < ef || score > (*res)[0].score {
				heap.Push(exp, candidate{id: nbrID, score: score})
				heap.Push(res, candidate{id: nbrID, score: score})
				if res.Len() > ef {
					heap.Pop(res) // evict worst to keep |res| == ef
				}
			}
		}
	}

	// Drain the min-heap into a slice sorted in DESCENDING score order.
	//
	// Draining in reverse index order achieves this without an extra sort:
	// heap.Pop(res) gives the minimum (worst) element each time.
	// Writing to indices len-1 … 0 therefore places:
	//   out[len-1] = worst,  out[0] = best   → descending. ✓
	out := make([]candidate, res.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(res).(candidate)
	}
	return out
}

// ──────────────────────────────────────────────────────────────────────────────
// selectNeighbors   (Algorithm 3 — simple variant, Malkov & Yashunin 2018)
// ──────────────────────────────────────────────────────────────────────────────

// selectNeighbors returns the best m candidates from an already-sorted
// (descending) candidate slice.
//
// This implements the "simple" neighbour selection policy from the paper.
// The heuristic variant (Algorithm 4) additionally favours diverse connections
// that span different parts of the space; it offers better recall at high
// compression ratios and can be added as a future enhancement.
func selectNeighbors(candidates []candidate, m int) []candidate {
	if len(candidates) <= m {
		return candidates
	}
	return candidates[:m]
}
