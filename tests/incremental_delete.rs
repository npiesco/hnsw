//! Behavioral regression guard for TRUE incremental delete on the const-generic
//! `Hnsw`: soft-delete must remove a node from search results WITHOUT degrading
//! the live result set, and deleting the navigation entry point must not break
//! the index.
//!
//! ## Why this is a real behavioral test (not a tripwire)
//!
//! Every assertion is driven through the REAL `insert` / `nearest` search path
//! over a seeded corpus. A naive post-filter tombstone (traverse then drop
//! deleted hits) fails two ways that this test pins:
//!   1. deleted node ids leak into `nearest` results, and
//!   2. the result set under-fills (deleted nodes consume the `ef`/result
//!      budget, so fewer than `k` live neighbors come back even though far more
//!      than `k` live nodes exist).
//! Deleting the entry point additionally breaks a naive implementation because
//! search seeds from a now-dead node. This test fails RED for those exact
//! runtime reasons and only goes GREEN once search is deletion-aware and the
//! entry point is maintained live.

use std::collections::BTreeSet;

use hnsw::{Hnsw, Searcher};
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
const N: usize = 400;
const M: usize = 12;
const M0: usize = 24;

fn corpus() -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(0xDE1E7E);
    (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect()
}

fn build(features: &[Vec<f32>]) -> Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> {
    let mut searcher = Searcher::default();
    let mut hnsw =
        Hnsw::new_params_and_prng(Euclidean, Default::default(), Pcg64::seed_from_u64(42));
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

#[test]
fn deleted_nodes_never_returned_and_entry_deletion_survives() {
    let features = corpus();
    let mut hnsw = build(&features);

    let ef = 64;
    let k = 20;

    // Sanity: before any deletion the index is real — a point queries itself as
    // the exact top-1 hit.
    let probe0 = 17usize;
    let pre = knn(&hnsw, &features[probe0], ef, k);
    assert_eq!(
        pre[0].index, probe0,
        "pre-delete: point {probe0} must be its own nearest neighbor"
    );

    // Deletion set: the live navigation entry point + a deterministic ~1/3
    // spread across the id space (guarantees multi-layer + hub deletions).
    let entry0 = hnsw.entry().expect("non-empty index has an entry point");
    let mut deleted: BTreeSet<usize> = BTreeSet::new();
    deleted.insert(entry0);
    for i in (0..N).step_by(3) {
        deleted.insert(i);
    }
    for &d in &deleted {
        hnsw.mark_delete(d);
    }

    assert_eq!(
        hnsw.live_count(),
        N - deleted.len(),
        "live_count must drop by exactly the number of distinct deleted nodes"
    );

    // Probes: every live corpus point's own feature + a few random queries.
    let mut probe_rng = Pcg64::seed_from_u64(7);
    let probes: Vec<Vec<f32>> = (0..N)
        .filter(|i| !deleted.contains(i))
        .map(|i| features[i].clone())
        .chain((0..50).map(|_| {
            (0..DIM)
                .map(|_| probe_rng.gen_range(-1.0f32..1.0f32))
                .collect()
        }))
        .collect();

    let live_total = N - deleted.len();
    assert!(live_total > k, "test needs more live nodes than k");

    for (qi, q) in probes.iter().enumerate() {
        let res = knn(&hnsw, q, ef, k);

        // 1. No deleted id ever leaks into results.
        for n in &res {
            assert!(
                !deleted.contains(&n.index),
                "probe {qi}: deleted node {} leaked into search results",
                n.index
            );
        }

        // 2. No under-fill: far more than k live nodes remain, so the search
        //    must return a full k live neighbors (deleted nodes must not consume
        //    the result budget).
        assert_eq!(
            res.len(),
            k,
            "probe {qi}: result set under-filled ({} < {k}) — deleted nodes \
             consumed the ef/result budget",
            res.len()
        );
    }

    // 3. The entry point was deleted; the index must have promoted a live
    //    replacement entry (never a tombstoned node).
    let new_entry = hnsw.entry().expect("index still has live nodes");
    assert!(
        !deleted.contains(&new_entry),
        "entry point must be a live node after the old entry was deleted"
    );
}
