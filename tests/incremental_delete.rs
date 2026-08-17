//! Behavioral regression guard for TRUE incremental delete on BOTH indexes:
//! soft-delete must remove a node from search results WITHOUT degrading the
//! live result set, and deleting the navigation entry point must not break the
//! index.
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
//!
//! Deleting the entry point additionally breaks a naive implementation because
//! search seeds from a now-dead node. This test fails RED for those exact
//! runtime reasons and only goes GREEN once search is deletion-aware and the
//! entry point is maintained live.
//!
//! `HnswRuntime` carries the identical guarantees: it is the runtime-degree
//! mirror of the const-generic index, so a tombstone must behave the same in
//! both. The runtime test below applies the same corpus, the same deletion set,
//! and the same assertions.

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

fn build_runtime(features: &[Vec<f32>]) -> HnswRuntime<Euclidean, Vec<f32>, Pcg64> {
    let mut searcher = Searcher::default();
    let mut hnsw = HnswRuntime::new_params_and_prng(
        Euclidean,
        M,
        M0,
        Default::default(),
        Pcg64::seed_from_u64(42),
    );
    for f in features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    hnsw
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

/// The runtime-degree index must honor tombstones exactly as the const-generic
/// one does: no deleted id may surface, the live result set must not under-fill,
/// and a deleted entry point must be replaced by a live one.
#[test]
fn runtime_deleted_nodes_never_returned_and_entry_deletion_survives() {
    let features = corpus();
    let mut hnsw = build_runtime(&features);

    let ef = 64;
    let k = 20;

    let probe0 = 17usize;
    let pre = knn_runtime(&hnsw, &features[probe0], ef, k);
    assert_eq!(
        pre[0].index, probe0,
        "pre-delete: point {probe0} must be its own nearest neighbor"
    );

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
        let res = knn_runtime(&hnsw, q, ef, k);

        for n in &res {
            assert!(
                !deleted.contains(&n.index),
                "probe {qi}: deleted node {} leaked into runtime search results",
                n.index
            );
        }

        assert_eq!(
            res.len(),
            k,
            "probe {qi}: runtime result set under-filled ({} < {k}) — deleted \
             nodes consumed the ef/result budget",
            res.len()
        );
    }

    let new_entry = hnsw.entry().expect("index still has live nodes");
    assert!(
        !deleted.contains(&new_entry),
        "runtime entry point must be a live node after the old entry was deleted"
    );
}

/// Soft-delete must not break the const/runtime mirror. Both indexes are built
/// from the same corpus with the same seed and degrees, the SAME ids are
/// tombstoned in both, and every probe must return the identical ranked list.
///
/// This is the assertion that catches a runtime tombstone implementation that
/// "works" in isolation but diverges from the const-generic one — e.g. filtering
/// results after the fact instead of excluding tombstones from the result heap
/// during traversal, which changes which live nodes get crowded out.
#[test]
fn runtime_and_const_agree_under_identical_deletions() {
    let features = corpus();
    let mut cst = build(&features);
    let mut rt = build_runtime(&features);

    let ef = 64;
    let k = 20;

    let mut deleted: BTreeSet<usize> = BTreeSet::new();
    deleted.insert(cst.entry().expect("non-empty index has an entry point"));
    for i in (0..N).step_by(3) {
        deleted.insert(i);
    }
    for &d in &deleted {
        cst.mark_delete(d);
        rt.mark_delete(d);
    }

    assert_eq!(
        cst.live_count(),
        rt.live_count(),
        "live_count must agree between const and runtime"
    );

    let mut probe_rng = Pcg64::seed_from_u64(7);
    let probes: Vec<Vec<f32>> = features
        .iter()
        .cloned()
        .chain((0..50).map(|_| {
            (0..DIM)
                .map(|_| probe_rng.gen_range(-1.0f32..1.0f32))
                .collect()
        }))
        .collect();

    for (qi, q) in probes.iter().enumerate() {
        let cst_res = knn(&cst, q, ef, k);
        let rt_res = knn_runtime(&rt, q, ef, k);
        assert_eq!(
            cst_res, rt_res,
            "runtime diverged from const under identical deletions on probe {qi}"
        );
    }
}
