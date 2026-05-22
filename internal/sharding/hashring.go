// Package sharding implements consistent hashing for distributing vector IDs
// across a cluster of nodes.
//
// Algorithm overview (Karger et al., 1997):
//
//  1. Each real node is mapped to `virtualNodes` points on a circular hash
//     ring using CRC-32 (IEEE polynomial).  Virtual nodes ensure that keys
//     distribute evenly even with a small number of real nodes.
//
//  2. To route a key, hash it with CRC-32, then walk clockwise on the ring
//     until the first virtual-node point is found.  The owning real node is
//     the one that virtual point belongs to.
//
//  3. When a node is added the ring is re-sorted in O(v log v) time where
//     v = total virtual nodes.  Key reassignment is proportional only to the
//     newly added node's share (~1/n of all keys).
//
// Thread safety: all public methods are protected by a RWMutex.
package sharding

import (
	"fmt"
	"hash/crc32"
	"sort"
	"sync"
)

const defaultVirtualNodes = 150 // virtual ring points per real node

// HashRing is a consistent-hashing ring.
type HashRing struct {
	mu           sync.RWMutex
	sortedHashes []uint32          // ring points in ascending order
	hashToNode   map[uint32]string // ring point → real node address
	virtualNodes int
}

// NewHashRing creates an empty ring.
// virtualNodes controls the number of ring points per real node;
// 150 is a good default that gives ≤ 5 % load imbalance for ≥ 5 nodes.
func NewHashRing(virtualNodes int) *HashRing {
	if virtualNodes <= 0 {
		virtualNodes = defaultVirtualNodes
	}
	return &HashRing{
		hashToNode:   make(map[uint32]string),
		virtualNodes: virtualNodes,
	}
}

// AddNode places a real node on the ring by creating virtualNodes evenly-spaced
// virtual points.  Each virtual point key is "<address>-<i>" so that the same
// address always maps to the same set of ring positions.
func (r *HashRing) AddNode(address string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i := 0; i < r.virtualNodes; i++ {
		h := crc32.ChecksumIEEE([]byte(fmt.Sprintf("%s-%d", address, i)))
		r.sortedHashes = append(r.sortedHashes, h)
		r.hashToNode[h] = address
	}

	// Keep the ring sorted so binary search works correctly.
	sort.Slice(r.sortedHashes, func(i, j int) bool {
		return r.sortedHashes[i] < r.sortedHashes[j]
	})
}

// RemoveNode removes all virtual points for the given address from the ring.
// Keys that were routed to this node will now route to the next clockwise node.
func (r *HashRing) RemoveNode(address string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Collect hashes that belong to this node.
	toRemove := make(map[uint32]bool)
	for i := 0; i < r.virtualNodes; i++ {
		h := crc32.ChecksumIEEE([]byte(fmt.Sprintf("%s-%d", address, i)))
		toRemove[h] = true
		delete(r.hashToNode, h)
	}

	// Rebuild sortedHashes without the removed points.
	filtered := r.sortedHashes[:0]
	for _, h := range r.sortedHashes {
		if !toRemove[h] {
			filtered = append(filtered, h)
		}
	}
	r.sortedHashes = filtered
}

// GetResponsibleNode returns the address of the node responsible for vectorID.
//
// The responsible node is the first real node encountered when walking
// clockwise from hash(vectorID) on the ring.  If the hash falls past the last
// ring point, we wrap around to the first point (ring topology).
//
// Returns "" if the ring is empty.
func (r *HashRing) GetResponsibleNode(vectorID string) string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.sortedHashes) == 0 {
		return ""
	}

	h := crc32.ChecksumIEEE([]byte(vectorID))

	// Binary search: find the first ring point ≥ h.
	idx := sort.Search(len(r.sortedHashes), func(i int) bool {
		return r.sortedHashes[i] >= h
	})

	// Wrap around if h exceeds all ring points.
	if idx >= len(r.sortedHashes) {
		idx = 0
	}

	return r.hashToNode[r.sortedHashes[idx]]
}

// Nodes returns the set of distinct real node addresses currently on the ring.
func (r *HashRing) Nodes() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	seen := make(map[string]bool, len(r.hashToNode)/r.virtualNodes+1)
	for _, addr := range r.hashToNode {
		seen[addr] = true
	}
	out := make([]string, 0, len(seen))
	for addr := range seen {
		out = append(out, addr)
	}
	return out
}
