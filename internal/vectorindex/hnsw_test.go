package vectorindex

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"testing"
)

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

// unitVec returns a unit-normalised float32 slice of length dim seeded by s.
func unitVec(dim int, s int64) []float32 {
	rng := rand.New(rand.NewSource(s))
	v := make([]float32, dim)
	var norm float64
	for i := range v {
		x := rng.Float32()*2 - 1
		v[i] = x
		norm += float64(x) * float64(x)
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}

// ──────────────────────────────────────────────────────────────────────────────
// Basic correctness
// ──────────────────────────────────────────────────────────────────────────────

// TestInsertAndSearch verifies that after inserting a set of known vectors,
// the Search RPC returns the exact inserted vector as the top-1 result when
// queried with itself.
func TestInsertAndSearch(t *testing.T) {
	const dim = 64
	g := NewHNSWGraph(8, 50)

	vectors := make(map[string][]float32, 200)
	for i := 0; i < 200; i++ {
		id := fmt.Sprintf("vec-%d", i)
		v := unitVec(dim, int64(i))
		vectors[id] = v
		g.Insert(id, v)
	}

	// Query with each vector's own embedding — must be the top-1 result.
	for id, v := range vectors {
		results := g.Search(v, 1)
		if len(results) == 0 {
			t.Errorf("Search(%s): got 0 results, want 1", id)
			continue
		}
		if results[0].ID != id {
			t.Errorf("Search(%s): top-1 = %s (score %.4f), want self",
				id, results[0].ID, results[0].Score)
		}
	}
}

// TestTopKOrdering verifies that results are returned in descending score order.
func TestTopKOrdering(t *testing.T) {
	const dim = 32
	g := NewHNSWGraph(8, 50)

	for i := 0; i < 100; i++ {
		g.Insert(fmt.Sprintf("v%d", i), unitVec(dim, int64(i+1000)))
	}

	query := unitVec(dim, 42)
	results := g.Search(query, 10)

	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Errorf("results not sorted: results[%d].Score=%.4f > results[%d].Score=%.4f",
				i, results[i].Score, i-1, results[i-1].Score)
		}
	}
}

// TestEmptyGraph verifies that Search on an empty graph returns nil.
func TestEmptyGraph(t *testing.T) {
	g := NewHNSWGraph(DefaultM, DefaultEfConstruction)
	if got := g.Search(unitVec(16, 1), 5); got != nil {
		t.Errorf("expected nil for empty graph, got %v", got)
	}
}

// TestSingleNode verifies correct behaviour with exactly one stored vector.
func TestSingleNode(t *testing.T) {
	g := NewHNSWGraph(DefaultM, DefaultEfConstruction)
	v := unitVec(16, 7)
	g.Insert("only", v)

	results := g.Search(v, 5) // ask for 5, should get 1
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != "only" {
		t.Errorf("wrong id: got %s, want 'only'", results[0].ID)
	}
}

// ──────────────────────────────────────────────────────────────────────────────
// Concurrency (run with -race to detect data races)
// ──────────────────────────────────────────────────────────────────────────────

// TestConcurrentInserts fires 8 goroutines each inserting 50 vectors and
// verifies that the graph is left in a navigable state (Search returns results).
func TestConcurrentInserts(t *testing.T) {
	const (
		dim        = 32
		goroutines = 8
		perRoutine = 50
	)

	g := NewHNSWGraph(8, 50)
	var wg sync.WaitGroup

	for gid := 0; gid < goroutines; gid++ {
		wg.Add(1)
		go func(gid int) {
			defer wg.Done()
			for i := 0; i < perRoutine; i++ {
				id := fmt.Sprintf("g%d-v%d", gid, i)
				seed := int64(gid*1000 + i)
				g.Insert(id, unitVec(dim, seed))
			}
		}(gid)
	}
	wg.Wait()

	// The graph must still answer queries correctly after concurrent writes.
	query := unitVec(dim, 9999)
	results := g.Search(query, 5)
	if len(results) == 0 {
		t.Fatal("Search returned 0 results after concurrent inserts")
	}
	// Scores must be in [−1, 1] (valid cosine range).
	for _, r := range results {
		if r.Score < -1.01 || r.Score > 1.01 {
			t.Errorf("score %.4f out of cosine range for id %s", r.Score, r.ID)
		}
	}
}

// TestConcurrentReadsAndWrites fires concurrent readers and writers
// simultaneously, the harshest test of the lock ordering discipline.
func TestConcurrentReadsAndWrites(t *testing.T) {
	const dim = 32
	g := NewHNSWGraph(8, 50)

	// Pre-populate with a few vectors so Search has something to traverse.
	for i := 0; i < 20; i++ {
		g.Insert(fmt.Sprintf("seed-%d", i), unitVec(dim, int64(i)))
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})

	// Writers: insert vectors continuously until stopped.
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				default:
					g.Insert(fmt.Sprintf("w%d-v%d", w, i), unitVec(dim, int64(w*10000+i)))
				}
			}
		}(w)
	}

	// Readers: search continuously until stopped.
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			q := unitVec(dim, int64(r+500))
			for {
				select {
				case <-stop:
					return
				default:
					g.Search(q, 5)
				}
			}
		}(r)
	}

	// Let the storm run for a short burst then stop cleanly.
	// With -race this will detect any concurrent map/slice access.
	// (In CI without the race detector this still catches panics and deadlocks.)
	var once sync.Once
	timer := make(chan struct{})
	go func() {
		// Run for ~50ms of goroutine-time; enough to exercise the lock paths.
		for i := 0; i < 500000; i++ {
		}
		once.Do(func() { close(timer) })
	}()
	<-timer
	close(stop)
	wg.Wait()
}
