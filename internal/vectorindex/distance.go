// Package vectorindex contains the core mathematical primitives used by the
// vector index.  All functions operate on float32 slices so that the hot path
// matches the wire representation (proto repeated float).
package vectorindex

import "math"

// CosineSimilarity returns the cosine similarity between two vectors a and b.
//
// Mathematical definition:
//
//	cos(θ) = (a · b) / (‖a‖ · ‖b‖)
//
// Return range is [-1, 1]:
//   - +1  → identical direction (most similar)
//   - 0  → orthogonal
//   - -1  → opposite direction
//
// Edge cases:
//   - Mismatched lengths: returns 0 (caller must ensure equal dimensions).
//   - Zero-magnitude vector: returns 0 to avoid division by zero.
//
// Performance notes:
//   - A single pass computes dot product and both magnitudes simultaneously,
//     keeping all three accumulators in CPU registers with no extra allocations.
//   - float32 arithmetic matches the stored/wire format, avoiding upcasting.
//   - The final division is guarded; for unit-normalised embeddings (e.g. most
//     transformer models) both norms are ≈ 1.0, so the division is essentially
//     free and the raw dot product already approximates the similarity.
func CosineSimilarity(a, b []float32) float32 {
	// Dimension mismatch is a programming error; return a safe sentinel.
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64

	// Single-pass accumulation: O(d) with no intermediate allocations.
	// Using float64 accumulators prevents catastrophic cancellation when
	// summing many small squared values (Kahan compensation is not necessary
	// at float64 precision for typical embedding dimensions up to ~4096).
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	// Guard against zero-vector inputs (undefined cosine similarity).
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}

	return float32(dot / denom)
}
