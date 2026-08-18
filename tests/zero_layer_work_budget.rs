//! The zero-layer search must reach a target recall within a work budget.
//!
//! ## The defect
//!
//! `search_single_layer` drains `Searcher.candidates`, which is a LIFO `Vec`
//! (`src/lib.rs`). That makes the zero-layer search DEPTH-FIRST with no
//! termination check: it expands whatever was pushed most recently and keeps
//! going until the frontier empties.
//!
//! The HNSW paper's Algorithm 2 uses a priority queue and stops when the nearest
//! remaining candidate is worse than the current worst result. This crate
//! already contains that implementation — `search_zero_layer_best_first`, added
//! for filtered search — but the ordinary path never used it.
//!
//! ## Why "different operating point" was the wrong reading
//!
//! Compared at equal `ef` the LIFO path looks BETTER, because it recalls more.
//! That comparison is misleading: `ef` is result-list capacity, not an expansion
//! budget, and the depth-first traversal spends far more work to get there. The
//! honest comparison is recall against measured distance evaluations.
//!
//! Measured on one graph, sweeping `ef` from 10 to 192 (recall@10, 50 queries):
//!
//! N=4000, dim=64:
//!
//! | evals | LIFO recall | evals | best-first recall |
//! | ---   | ---         | ---   | ---               |
//! | 1223  | 0.8460      | 1131  | 0.9140            |
//! | 1611  | 0.9100      | 1487  | 0.9640            |
//! | 1909  | 0.9640      | 1782  | 0.9900            |
//! | 2369  | 0.9800      | 2241  | 0.9960            |
//!
//! N=20000, dim=128:
//!
//! | evals | LIFO recall | evals | best-first recall |
//! | ---   | ---         | ---   | ---               |
//! | 3154  | 0.6940      | 2573  | 0.7700            |
//! | 4323  | 0.7800      | 3542  | 0.8420            |
//!
//! Best-first reaches HIGHER recall for LESS work at every comparable point on
//! both curves. It is not a different operating point; it strictly dominates.
//!
//! ## What this test asserts
//!
//! A budget the depth-first traversal cannot meet and the best-first traversal
//! can, stated in distance evaluations so it is machine-independent and
//! deterministic rather than a timing.

use hnsw::{Hnsw, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};
use std::cell::Cell;

thread_local! {
    /// Thread-local, not a global atomic: cargo runs tests concurrently and a
    /// shared counter is summed across them.
    static EVALS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

impl Metric<Vec<f32>> for Counting {
    type Unit = u64;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u64 {
        EVALS.with(|e| e.set(e.get() + 1));
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits() as u64
    }
}

const N: usize = 20_000;
const DIM: usize = 128;
const K: usize = 10;

type Idx = Hnsw<Counting, Vec<f32>, Pcg64, 12, 24>;

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

fn build(features: &[Vec<f32>]) -> Idx {
    let mut searcher = Searcher::default();
    let mut hnsw: Idx = Hnsw::new(Counting);
    for f in features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    hnsw
}

/// Recall and evaluations for `nearest` at a given `ef`.
fn measure(hnsw: &Idx, features: &[Vec<f32>], qs: &[Vec<f32>], ef: usize) -> (f64, f64) {
    let truths: Vec<std::collections::BTreeSet<usize>> = qs
        .iter()
        .map(|q| {
            let mut all: Vec<(u64, usize)> = features
                .iter()
                .enumerate()
                .map(|(i, f)| (Counting.distance(q, f), i))
                .collect();
            all.sort();
            all.into_iter().take(K).map(|(_, i)| i).collect()
        })
        .collect();

    EVALS.with(|e| e.set(0));
    let mut searcher = Searcher::default();
    let mut hits = 0usize;
    for (q, truth) in qs.iter().zip(&truths) {
        let mut dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            K
        ];
        let got = hnsw.nearest(q, ef, &mut searcher, &mut dest);
        hits += got.iter().filter(|n| truth.contains(&n.index)).count();
    }
    let evals = EVALS.with(|e| e.get()) as f64 / qs.len() as f64;
    (hits as f64 / (K * qs.len()) as f64, evals)
}

/// THE assertion. Both numbers come from the measurement above.
///
/// Best-first reaches 0.7700 in 2573 evaluations at this size; the depth-first
/// traversal needs 4323 to reach 0.7800 and only manages 0.6940 by 3154. The
/// budget sits between those regimes with margin on both sides.
#[test]
fn the_zero_layer_search_reaches_target_recall_within_budget() {
    /// Predeclared, both derived from the sweep rather than chosen.
    const TARGET_RECALL: f64 = 0.75;
    const MAX_EVALS: f64 = 3_400.0;

    let features = corpus(N, DIM, 42);
    let hnsw = build(&features);
    let qs = corpus(50, DIM, 0xC0FFEE);

    // Find the cheapest `ef` that reaches the target, and report its cost.
    let mut best: Option<(usize, f64, f64)> = None;
    for ef in [32usize, 48, 64, 96, 128, 192, 256] {
        let (recall, evals) = measure(&hnsw, &features, &qs, ef);
        if recall >= TARGET_RECALL {
            best = Some((ef, recall, evals));
            break;
        }
    }

    let (ef, recall, evals) = best.unwrap_or_else(|| {
        panic!(
            "no `ef` up to 256 reached recall {TARGET_RECALL} on a {N}-vector \
             index; the zero-layer search is not converging at all"
        )
    });

    assert!(
        evals <= MAX_EVALS,
        "reaching recall {TARGET_RECALL} needed ef={ef} and {evals:.0} distance \
         evaluations per query, over the {MAX_EVALS:.0} budget (achieved \
         {recall:.4}). The zero-layer search is draining a LIFO `Vec` with no \
         early stop, so it explores depth-first and pays far more work per unit \
         of recall than the priority-queue traversal the paper specifies — which \
         this crate already implements for filtered search."
    );
}

/// The change must not cost correctness: an exact match must still come back
/// first, and results must stay ordered.
#[test]
fn results_remain_correct_and_ordered() {
    let features = corpus(2_000, 32, 7);
    let hnsw = build(&features);
    let mut searcher = Searcher::default();

    for probe in [0usize, 500, 1_999] {
        let mut dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            K
        ];
        let got = hnsw.nearest(&features[probe], 64, &mut searcher, &mut dest);

        assert_eq!(
            got.first().map(|n| n.index),
            Some(probe),
            "querying with an indexed vector must return it first"
        );
        for w in got.windows(2) {
            assert!(
                w[0].distance <= w[1].distance,
                "results are not ordered by distance"
            );
        }
        assert_eq!(got.len(), K, "result set under-filled");
    }
}
