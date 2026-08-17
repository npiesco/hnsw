//! Recall floor for a long-lived, incrementally-deleted index — and the pin for
//! the navigation-seed policy.
//!
//! `mark_delete` exists for indexes that live a long time and accumulate
//! tombstones, so the guarantee that matters is: **stripping the top of the
//! hierarchy must not collapse recall.** Repeatedly deleting the live entry
//! point walks the towers top-down, so it eats the upper layers first — the
//! worst case for navigation.
//!
//! ## Why this also pins the seed policy
//!
//! The internal navigation seed (`initialize_searcher` / `entry_feature`) is
//! deliberately NOT deletion-aware, unlike the public `entry()`. That asymmetry
//! looks like an oversight, so it invites a "fix". It was measured:
//!
//! | top-of-hierarchy kills | seed as-is | first-LIVE-top-layer-node seed |
//! |-----------------------:|-----------:|-------------------------------:|
//! | 0                      |   0.9735   |             0.9735             |
//! | 1                      |   0.9735   |             0.9723             |
//! | 3                      |   0.9735   |             0.9735             |
//! | 8                      |   0.9710   |             0.9648             |
//! | 20                     |   0.9698   |             0.9585             |
//!
//! (recall@10, ef=24, 8 clusters x 50 pts, 40 trials, identical for both indexes.)
//!
//! A deletion-aware seed was never better and up to ~1.1pp worse. A tombstone is
//! still a fine *navigational* waypoint — its feature and edges are intact, and
//! the upper-layer descent never consults `is_deleted` — whereas jumping to a
//! different top-layer node lands the greedy (`cap == 1`) descent in a worse
//! basin. What actually mattered was that a dead seed must not hold a slot in the
//! bounded result heap; that is pinned by `tests/tombstone_seed_underfill.rs`.
//!
//! The threshold below is set so the deletion-aware variant (drop ~0.015) fails
//! while current behavior (drop ~0.004) passes.

use std::collections::BTreeSet;

use hnsw::{Hnsw, HnswRuntime, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

struct Euclidean;

impl Metric<Vec<f32>> for Euclidean {
    type Unit = u64;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits() as u64
    }
}

const DIM: usize = 8;
const CLUSTERS: usize = 8;
const PER_CLUSTER: usize = 50;
const N: usize = CLUSTERS * PER_CLUSTER;
const M: usize = 12;
const M0: usize = 24;

const EF: usize = 24;
const K: usize = 10;
const TRIALS: u64 = 12;
const QUERIES: usize = 20;
/// Deep enough to strip the upper layers several times over.
const TOP_KILLS: usize = 20;
/// Max tolerated recall loss from stripping the top of the hierarchy.
const MAX_RECALL_DROP: f64 = 0.01;

/// Clustered corpus — dense clusters are where navigation quality shows.
fn corpus(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let centers: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-10.0f32..10.0f32)).collect())
        .collect();
    let mut out = Vec::with_capacity(N);
    for c in &centers {
        for _ in 0..PER_CLUSTER {
            out.push(
                c.iter()
                    .map(|&x| x + rng.gen_range(-1.0f32..1.0f32))
                    .collect(),
            );
        }
    }
    out
}

fn build(features: &[Vec<f32>], seed: u64) -> Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> {
    let mut searcher = Searcher::default();
    let mut hnsw =
        Hnsw::new_params_and_prng(Euclidean, Default::default(), Pcg64::seed_from_u64(seed));
    for f in features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    hnsw
}

fn build_runtime(features: &[Vec<f32>], seed: u64) -> HnswRuntime<Euclidean, Vec<f32>, Pcg64> {
    let mut searcher = Searcher::default();
    let mut hnsw = HnswRuntime::new_params_and_prng(
        Euclidean,
        M,
        M0,
        Default::default(),
        Pcg64::seed_from_u64(seed),
    );
    for f in features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    hnsw
}

fn knn(
    hnsw: &Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0>,
    q: &[f32],
    ef: usize,
    k: usize,
) -> Vec<Neighbor<u64>> {
    let mut searcher = Searcher::default();
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ];
    hnsw.nearest(&q.to_vec(), ef, &mut searcher, &mut dest)
        .to_vec()
}

fn knn_runtime(
    hnsw: &HnswRuntime<Euclidean, Vec<f32>, Pcg64>,
    q: &[f32],
    ef: usize,
    k: usize,
) -> Vec<Neighbor<u64>> {
    let mut searcher = Searcher::default();
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ];
    hnsw.nearest(&q.to_vec(), ef, &mut searcher, &mut dest)
        .to_vec()
}

/// Exact top-k over live nodes only.
fn ground_truth(
    features: &[Vec<f32>],
    deleted: &BTreeSet<usize>,
    q: &[f32],
    k: usize,
) -> Vec<usize> {
    let met = Euclidean;
    let qv = q.to_vec();
    let mut all: Vec<(u64, usize)> = (0..features.len())
        .filter(|i| !deleted.contains(i))
        .map(|i| (met.distance(&qv, &features[i]), i))
        .collect();
    all.sort();
    all.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Queries land near live corpus points, i.e. inside dense clusters, rather than
/// in empty space where every method trivially agrees.
fn queries(features: &[Vec<f32>], deleted: &BTreeSet<usize>, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(0xC0FFEE + seed);
    (0..QUERIES)
        .map(|_| {
            let base = loop {
                let i = rng.gen_range(0..N);
                if !deleted.contains(&i) {
                    break i;
                }
            };
            features[base]
                .iter()
                .map(|&x| x + rng.gen_range(-0.25f32..0.25f32))
                .collect()
        })
        .collect()
}

struct Score {
    hits: usize,
    total: usize,
    underfilled: usize,
}

impl Score {
    fn new() -> Self {
        Self {
            hits: 0,
            total: 0,
            underfilled: 0,
        }
    }

    fn recall(&self) -> f64 {
        self.hits as f64 / self.total as f64
    }
}

/// Recall for both indexes after deleting `top_kills` successive entry points.
fn measure(top_kills: usize) -> (Score, Score) {
    let mut cst = Score::new();
    let mut rt = Score::new();

    for seed in 0..TRIALS {
        let features = corpus(seed);
        let mut hnsw = build(&features, seed);
        let mut runtime = build_runtime(&features, seed);

        let mut deleted = BTreeSet::new();
        for _ in 0..top_kills {
            match hnsw.entry() {
                Some(e) => {
                    hnsw.mark_delete(e);
                    runtime.mark_delete(e);
                    deleted.insert(e);
                }
                None => break,
            }
        }

        for q in &queries(&features, &deleted, seed) {
            let truth: BTreeSet<usize> = ground_truth(&features, &deleted, q, K)
                .into_iter()
                .collect();

            let got = knn(&hnsw, q, EF, K);
            if got.len() < K {
                cst.underfilled += 1;
            }
            cst.hits += got.iter().filter(|n| truth.contains(&n.index)).count();
            cst.total += truth.len();

            let got_rt = knn_runtime(&runtime, q, EF, K);
            if got_rt.len() < K {
                rt.underfilled += 1;
            }
            rt.hits += got_rt.iter().filter(|n| truth.contains(&n.index)).count();
            rt.total += truth.len();
        }
    }

    (cst, rt)
}

#[test]
fn stripping_the_hierarchy_top_does_not_collapse_recall() {
    let (base_cst, base_rt) = measure(0);
    let (kill_cst, kill_rt) = measure(TOP_KILLS);

    for (label, base, killed) in [
        ("const", &base_cst, &kill_cst),
        ("runtime", &base_rt, &kill_rt),
    ] {
        // Far more than K live nodes remain, so a short result set can only mean
        // tombstones ate the budget.
        assert_eq!(
            base.underfilled, 0,
            "{label}: baseline under-filled with no deletions at all"
        );
        assert_eq!(
            killed.underfilled, 0,
            "{label}: result set under-filled after deleting {TOP_KILLS} entry \
             points — tombstones are consuming the result budget"
        );

        let drop = base.recall() - killed.recall();
        assert!(
            drop <= MAX_RECALL_DROP,
            "{label}: recall@{K} fell {drop:.4} (from {:.4} to {:.4}) after \
             deleting the top {TOP_KILLS} entry points, over the \
             {MAX_RECALL_DROP} budget. Navigating from a tombstoned seed should \
             be free — a dead node keeps its feature and its edges. A regression \
             here usually means the seed policy was changed to prefer a live node \
             over the canonical entry point, which measures WORSE (see the module \
             docs).",
            base.recall(),
            killed.recall()
        );
    }
}
