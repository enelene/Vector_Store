// Command benchmarker is a load-testing and accuracy-measurement client for the
// VectorDB gRPC server.
//
// It runs three phases against a running server:
//
//  1. Ingestion  — inserts a synthetic dataset of high-dimensional vectors and
//     reports total time and throughput (inserts/sec).
//  2. Throughput — issues many concurrent Search queries and reports QPS plus
//     latency percentiles (p50/p90/p99/max).
//  3. Accuracy   — measures Recall@K of the HNSW index against an EXACT
//     brute-force baseline computed locally.
//
// Design goal — the CLIENT must not be the bottleneck:
//   - Concurrency comes from a configurable worker pool (--workers).
//   - Workers spread their RPCs across a pool of gRPC connections (--conns) so a
//     single HTTP/2 connection's max-concurrent-streams limit never caps load.
//   - Connections are warmed up before the timed phases so dial latency does not
//     pollute the measurements.
//
// Only the standard library, gRPC and this project's own packages are used.
// The brute-force baseline reuses the server's exact cosine metric
// (vectorindex.CosineSimilarity) so "ground truth" is defined by the same
// distance function the index approximates.
//
// Example:
//
//	go run ./cmd/benchmarker --addr=localhost:50051 --n=10000 --queries=5000 --topk=10
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	pb "vectordb/api/proto"
	"vectordb/internal/vectorindex"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	addr := flag.String("addr", "localhost:50051", "server gRPC address")
	dim := flag.Int("dim", 128, "vector dimensionality (SIFT1M uses 128)")
	n := flag.Int("n", 10000, "number of vectors to insert")
	clusters := flag.Int("clusters", 100, "latent clusters in the synthetic data (gives realistic neighbourhoods)")
	noise := flag.Float64("noise", 0.05, "per-coordinate gaussian noise around a cluster centre")
	queries := flag.Int("queries", 5000, "number of Search RPCs for the QPS measurement")
	topK := flag.Int("topk", 10, "neighbours requested per query (the K in Recall@K)")
	recallQ := flag.Int("recall-queries", 200, "queries used for the (expensive) recall measurement")
	workers := flag.Int("workers", 50, "concurrent worker goroutines")
	conns := flag.Int("conns", 8, "gRPC connections shared round-robin by the workers")
	settle := flag.Duration("settle", 2*time.Second, "pause after ingestion so replication/apply can catch up")
	seed := flag.Int64("seed", 42, "PRNG seed for reproducible datasets")
	flag.Parse()

	if *clusters < 1 {
		*clusters = 1
	}
	if *conns < 1 {
		*conns = 1
	}
	k := *topK
	if k > *n {
		k = *n
	}

	// ── Generate the dataset and a pool of query vectors ─────────────────────
	// Clustered data (centre + gaussian noise) mimics real embeddings, where
	// each point has well-defined near neighbours — so Recall@K is meaningful.
	rng := rand.New(rand.NewSource(*seed))
	log.Printf("generating data: n=%d dim=%d clusters=%d noise=%.3f", *n, *dim, *clusters, *noise)
	centers := randomVectors(rng, *clusters, *dim)
	dataset := clusteredVectors(rng, *n, *dim, centers, *noise)

	queryPoolSize := *recallQ
	if queryPoolSize < 1000 {
		queryPoolSize = 1000 // enough distinct queries to keep the QPS phase varied
	}
	queryPool := clusteredVectors(rng, queryPoolSize, *dim, centers, *noise)

	// ── Connect (pool of gRPC connections) ───────────────────────────────────
	clients := make([]pb.VectorDatabaseClient, *conns)
	for i := range clients {
		cc, err := grpc.NewClient(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			log.Fatalf("dial %s: %v", *addr, err)
		}
		defer cc.Close()
		clients[i] = pb.NewVectorDatabaseClient(cc)
	}

	// Warm up every connection (Search is read-only, so it does not pollute the
	// dataset) so HTTP/2 dial cost is excluded from the timed phases.
	for _, c := range clients {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, _ = c.Search(ctx, &pb.SearchRequest{QueryVector: queryPool[0], TopK: 1})
		cancel()
	}

	// ── Phase 1: ingestion ───────────────────────────────────────────────────
	log.Printf("ingestion: inserting %d vectors via %d workers over %d connections", *n, *workers, *conns)
	var insErr int64
	start := time.Now()
	runPool(*workers, *n, func(wid, j int) {
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		_, err := clients[wid%*conns].Insert(ctx, &pb.InsertRequest{
			Id:     fmt.Sprintf("vec-%d", j),
			Vector: dataset[j],
		})
		if err != nil {
			atomic.AddInt64(&insErr, 1)
		}
	})
	ingest := time.Since(start)
	insOK := *n - int(insErr)
	ingestRate := ratePerSec(insOK, ingest)

	// ── Settle ───────────────────────────────────────────────────────────────
	if *settle > 0 {
		log.Printf("settling for %s (let replication/apply catch up)…", *settle)
		time.Sleep(*settle)
	}

	// ── Phase 2: throughput (QPS) ────────────────────────────────────────────
	log.Printf("throughput: issuing %d Search queries (top-%d) via %d workers", *queries, k, *workers)
	lat := make([]time.Duration, *queries) // lat[j] written only by the worker handling job j → no race
	var qErr int64
	qStart := time.Now()
	runPool(*workers, *queries, func(wid, j int) {
		q := queryPool[j%len(queryPool)]
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		t0 := time.Now()
		_, err := clients[wid%*conns].Search(ctx, &pb.SearchRequest{QueryVector: q, TopK: int32(k)})
		lat[j] = time.Since(t0)
		if err != nil {
			atomic.AddInt64(&qErr, 1)
		}
	})
	qElapsed := time.Since(qStart)
	qOK := *queries - int(qErr)
	qps := ratePerSec(qOK, qElapsed)
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })

	// ── Phase 3: accuracy (Recall@K) ─────────────────────────────────────────
	// Recall@1 isolates "does the index find THE nearest neighbour" (a check on
	// correctness); Recall@K is the harder "did it find all K nearest" metric.
	log.Printf("accuracy: measuring Recall@1 and Recall@%d over %d queries (brute-force baseline)…", k, *recallQ)
	recall1 := measureRecall(clients, *conns, dataset, queryPool[:*recallQ], 1, *workers)
	recallK := measureRecall(clients, *conns, dataset, queryPool[:*recallQ], k, *workers)

	// ── Report ───────────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("════════════════════════ BENCHMARK REPORT ════════════════════════")
	fmt.Printf("server             : %s\n", *addr)
	fmt.Printf("dataset            : n=%d dim=%d clusters=%d noise=%.3f seed=%d\n", *n, *dim, *clusters, *noise, *seed)
	fmt.Printf("concurrency        : workers=%d conns=%d\n", *workers, *conns)
	fmt.Println("───────────────────────────────────────────────────────────────────")
	fmt.Printf("ingestion          : %d ok / %d err in %s\n", insOK, insErr, ingest.Round(time.Millisecond))
	fmt.Printf("ingestion rate     : %.0f inserts/sec\n", ingestRate)
	fmt.Println("───────────────────────────────────────────────────────────────────")
	fmt.Printf("search queries     : %d ok / %d err in %s\n", qOK, qErr, qElapsed.Round(time.Millisecond))
	fmt.Printf("throughput (QPS)   : %.0f queries/sec\n", qps)
	fmt.Printf("latency p50/p90/p99/max : %s / %s / %s / %s\n",
		percentile(lat, 50).Round(time.Microsecond),
		percentile(lat, 90).Round(time.Microsecond),
		percentile(lat, 99).Round(time.Microsecond),
		percentile(lat, 100).Round(time.Microsecond))
	fmt.Println("───────────────────────────────────────────────────────────────────")
	fmt.Printf("Recall@1           : %.4f  (%d queries vs exact brute-force)\n", recall1, *recallQ)
	if k > 1 {
		fmt.Printf("Recall@%-2d          : %.4f  (%d queries vs exact brute-force)\n", k, recallK, *recallQ)
	}
	fmt.Println("═══════════════════════════════════════════════════════════════════")
}

// ── Worker pool ───────────────────────────────────────────────────────────────

// runPool runs fn for every jobIdx in [0,jobs) across exactly `workers`
// goroutines. The job index is fed through a buffered channel; each worker is
// passed a stable workerID so it can pick a connection deterministically.
func runPool(workers, jobs int, fn func(workerID, jobIdx int)) {
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	ch := make(chan int, workers*2)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := range ch {
				fn(id, j)
			}
		}(w)
	}
	for j := 0; j < jobs; j++ {
		ch <- j
	}
	close(ch)
	wg.Wait()
}

// ── Accuracy ──────────────────────────────────────────────────────────────────

// measureRecall returns the mean Recall@k over the given queries:
//
//	Recall@k = (1/|Q|) · Σ_q |serverTopK(q) ∩ exactTopK(q)| / k
//
// The exact top-k (ground truth) is computed by brute force locally; the
// server's top-k comes from the HNSW index. Both the brute force and the RPCs
// run across the worker pool so this expensive phase still uses all cores.
func measureRecall(
	clients []pb.VectorDatabaseClient, conns int,
	dataset, queries [][]float32, k, workers int,
) float64 {
	if len(queries) == 0 || k == 0 {
		return 0
	}
	var hits int64
	runPool(workers, len(queries), func(wid, j int) {
		q := queries[j]

		// Ground truth: exact nearest neighbours by cosine similarity.
		truth := topKBrute(q, dataset, k)
		truthSet := make(map[string]struct{}, k)
		for _, idx := range truth {
			truthSet[fmt.Sprintf("vec-%d", idx)] = struct{}{}
		}

		// Approximate result from the server's HNSW index.
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		resp, err := clients[wid%conns].Search(ctx, &pb.SearchRequest{QueryVector: q, TopK: int32(k)})
		if err != nil {
			return
		}
		for _, r := range resp.Results {
			if _, ok := truthSet[r.Id]; ok {
				atomic.AddInt64(&hits, 1)
			}
		}
	})
	return float64(hits) / float64(len(queries)*k)
}

// topKBrute returns the indices of the k highest-cosine-similarity vectors in
// data relative to q (exact, O(n) scan + sort).
func topKBrute(q []float32, data [][]float32, k int) []int {
	type sc struct {
		idx int
		s   float32
	}
	scores := make([]sc, len(data))
	for i := range data {
		scores[i] = sc{i, vectorindex.CosineSimilarity(q, data[i])}
	}
	sort.Slice(scores, func(a, b int) bool { return scores[a].s > scores[b].s })
	if k > len(scores) {
		k = len(scores)
	}
	out := make([]int, k)
	for i := 0; i < k; i++ {
		out[i] = scores[i].idx
	}
	return out
}

// ── Synthetic data ────────────────────────────────────────────────────────────

// randomVectors returns count unit-normalised gaussian vectors (cluster centres).
func randomVectors(rng *rand.Rand, count, dim int) [][]float32 {
	out := make([][]float32, count)
	for i := range out {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		normalize(v)
		out[i] = v
	}
	return out
}

// clusteredVectors returns count unit-normalised vectors, each a random cluster
// centre perturbed by per-coordinate gaussian noise.
func clusteredVectors(rng *rand.Rand, count, dim int, centers [][]float32, noise float64) [][]float32 {
	out := make([][]float32, count)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		v := make([]float32, dim)
		for d := range v {
			v[d] = c[d] + float32(rng.NormFloat64()*noise)
		}
		normalize(v)
		out[i] = v
	}
	return out
}

// normalize scales v to unit L2 length in place (no-op for the zero vector).
func normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// ── Stats helpers ─────────────────────────────────────────────────────────────

// ratePerSec returns count/elapsed in per-second units, guarding against a
// zero-duration divide.
func ratePerSec(count int, elapsed time.Duration) float64 {
	if elapsed <= 0 {
		return 0
	}
	return float64(count) / elapsed.Seconds()
}

// percentile returns the p-th percentile (0–100) of an ASCENDING-sorted slice
// using the nearest-rank method.
func percentile(sorted []time.Duration, p float64) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(p/100*float64(len(sorted)))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
