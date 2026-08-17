//! Cross-cluster reachability: a dense cluster must not evict the links that
//! reach everything outside it.
//!
//! ## The failure this pins
//!
//! Neighbor lists are finite. When a node's list is full and a new candidate
//! arrives, the naive rule is "keep the M nearest" — evict whichever current
//! neighbor is farthest. That rule is stable under uniform data and pathological
//! under clustered data: every member of a dense cluster is nearer to every
//! other member than to anything outside it, so each one's list saturates with
//! its own cluster and every outbound link is pruned. The cluster becomes a
//! closed component. A search entering it can never leave, and points outside
//! become unreachable no matter how large `ef` is.
//!
//! The HNSW paper's neighbor-selection heuristic (Algorithm 4) exists for this:
//! it keeps a candidate that is closer to the base node than to any already-kept
//! neighbor, which preserves one link per direction instead of M links into the
//! nearest blob.
//!
//! ## Why these are real tests
//!
//! Each drives the real `insert`/`nearest` path over a real graph and asserts on
//! returned distances. They fail RED against nearest-M truncation for the actual
//! runtime reason — the correct answers are absent from the result — and they
//! are `ef`-independent and insertion-order-independent, so neither a larger
//! candidate pool nor a luckier ordering can mask a regression.

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

fn decode(unit: u64) -> f32 {
    f32::from_bits(unit as u32)
}

/// Three points beside the origin, then a dense cluster far away from both the
/// origin and them. Querying the origin must return the three near points.
fn near_then_far(ef: usize, near_first: bool) -> Vec<f32> {
    let mut searcher: Searcher<u64> = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 16, 32> = Hnsw::new(Euclidean);

    fn insert_near(h: &mut Hnsw<Euclidean, Vec<f32>, Pcg64, 16, 32>, s: &mut Searcher<u64>) {
        for c in [1.0f32, 2.0, 3.0] {
            h.insert(vec![c, 0.0], s);
        }
    }
    fn insert_far(h: &mut Hnsw<Euclidean, Vec<f32>, Pcg64, 16, 32>, s: &mut Searcher<u64>) {
        for i in 0..37 {
            h.insert(vec![500.0 + i as f32, 500.0], s);
        }
    }

    if near_first {
        insert_near(&mut hnsw, &mut searcher);
        insert_far(&mut hnsw, &mut searcher);
    } else {
        insert_far(&mut hnsw, &mut searcher);
        insert_near(&mut hnsw, &mut searcher);
    }

    let mut dest = [Neighbor {
        index: !0,
        distance: !0,
    }; 3];
    hnsw.nearest(&vec![0.0, 0.0], ef, &mut searcher, &mut dest)
        .iter()
        .map(|n| decode(n.distance))
        .collect()
}

#[test]
fn a_dense_cluster_does_not_hide_points_outside_it() {
    let got = near_then_far(200, true);
    assert_eq!(got.len(), 3);
    for d in &got {
        assert!(
            *d < 10.0,
            "expected the three points beside the origin (distances 1, 2, 3), \
             got {:?} — the far cluster pruned every link reaching them",
            got
        );
    }
}

#[test]
fn cross_cluster_reachability_does_not_depend_on_insertion_order() {
    for near_first in [true, false] {
        let got = near_then_far(200, near_first);
        for d in &got {
            assert!(
                *d < 10.0,
                "near_first={}: got {:?}, expected the near points",
                near_first,
                got
            );
        }
    }
}

#[test]
fn cross_cluster_reachability_does_not_depend_on_a_large_candidate_pool() {
    // A bigger `ef` explores more of the graph, but it cannot traverse an edge
    // that was never kept. If this only passes at high `ef`, pruning is broken.
    for ef in [4, 16, 64, 200, 1000] {
        let got = near_then_far(ef, true);
        for d in &got {
            assert!(
                *d < 10.0,
                "ef={}: got {:?}, expected the near points",
                ef,
                got
            );
        }
    }
}

#[test]
fn every_point_of_a_multi_cluster_corpus_finds_itself() {
    // Four tight, well-separated clusters. Probing any stored point with its own
    // vector must return that point at distance zero; a closed component shows
    // up here as a point that cannot find itself from the global entry.
    let mut rng = Pcg64::seed_from_u64(42);
    let mut searcher: Searcher<u64> = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 16, 32> = Hnsw::new(Euclidean);

    let centres = [
        [0.0f32, 0.0],
        [1000.0, 0.0],
        [0.0, 1000.0],
        [1000.0, 1000.0],
    ];
    let mut stored = Vec::new();
    for centre in centres {
        for _ in 0..60 {
            let v = vec![
                centre[0] + rng.gen_range(-1.0..1.0),
                centre[1] + rng.gen_range(-1.0..1.0),
            ];
            hnsw.insert(v.clone(), &mut searcher);
            stored.push(v);
        }
    }

    for v in &stored {
        let mut dest = [Neighbor {
            index: !0,
            distance: !0,
        }; 1];
        let res = hnsw.nearest(v, 64, &mut searcher, &mut dest);
        assert_eq!(res.len(), 1);
        assert!(
            decode(res[0].distance) < 1e-2,
            "a stored point could not find itself: distance {}",
            decode(res[0].distance)
        );
    }
}

#[test]
fn the_runtime_index_has_the_same_cross_cluster_reachability() {
    let mut searcher: Searcher<u64> = Searcher::default();
    let mut hnsw = HnswRuntime::<Euclidean, Vec<f32>, Pcg64>::new(Euclidean, 16, 32);

    for c in [1.0f32, 2.0, 3.0] {
        hnsw.insert(vec![c, 0.0], &mut searcher);
    }
    for i in 0..37 {
        hnsw.insert(vec![500.0 + i as f32, 500.0], &mut searcher);
    }

    let mut dest = [Neighbor {
        index: !0,
        distance: !0,
    }; 3];
    let got: Vec<f32> = hnsw
        .nearest(&vec![0.0, 0.0], 200, &mut searcher, &mut dest)
        .iter()
        .map(|n| decode(n.distance))
        .collect();

    assert_eq!(got.len(), 3);
    for d in &got {
        assert!(*d < 10.0, "runtime index: got {:?}", got);
    }
}
