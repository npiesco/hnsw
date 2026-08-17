//! Regression guard: a soft-deleted **navigation seed** must not consume a slot
//! in the bounded result heap.
//!
//! ## The bug this pins
//!
//! Two tombstone guards exist on the search path, and both only catch nodes
//! discovered *during traversal*:
//!
//!   * `search_single_layer` skips a deleted node at the zero layer before it
//!     can enter `nearest` (it still goes into `candidates`, preserving
//!     connectivity), and
//!   * a `retain` scrubs `nearest` *after* `search_zero_layer` returns.
//!
//! The seed never passes through either guard on the way in. `initialize_searcher`
//! pushes the entry point straight into `searcher.nearest`, and `lower_search`
//! re-pushes the carried best candidate into `nearest` at every descent. Neither
//! checks `is_deleted`. So the zero-layer search can *begin* with a tombstone
//! already sitting in `nearest`.
//!
//! The trailing `retain` does remove it — but only after the search has finished.
//! For the entire duration of that search the tombstone occupied one of the `cap`
//! slots in the bounded heap, and the insert logic treats the pool as full at
//! `nearest.len() == cap` and evicts the worst. So a live neighbor that belonged
//! in the result set was rejected or displaced to make room for a dead node.
//! Then `retain` drops the corpse and the caller gets back `cap - 1`.
//!
//! Signature: under-fill by *exactly one*, only at zero slack
//! (`ef == wanted == live_count`), only when the deleted node is the seed.
//!
//! ## Why the existing delete test does not catch it
//!
//! `tests/incremental_delete.rs` queries with `ef = 64, k = 20`. That 44-slot
//! slack absorbs the stolen slot, so the under-fill is invisible. The bug is only
//! observable when the budget is exact.
//!
//! ## Why this is a real behavioral test
//!
//! Every assertion runs through the real `insert` / `mark_delete` / `nearest`
//! path. The control case — same corpus, same zero-slack budget, no deletions —
//! is asserted to fill completely, which proves a short result is caused by the
//! tombstone accounting and not by approximate search failing to reach the nodes.

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

const DIM: usize = 4;
const N: usize = 64;
const M: usize = 12;
const M0: usize = 24;
const TRIALS: u64 = 30;

fn corpus(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect()
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

/// Control: with no deletions at all and the same zero-slack budget
/// (`ef == k == len`), the search must return a completely full result set.
///
/// This is what makes the failing cases below meaningful. If approximate search
/// simply could not reach every node at this budget, this control would fail
/// too, and a short result would prove nothing about tombstones.
#[test]
fn control_zero_slack_fills_completely_without_deletions() {
    for seed in 0..TRIALS {
        let features = corpus(seed);
        let hnsw = build(&features, seed);
        let rt = build_runtime(&features, seed);

        // Query at the seed node's position — the same probe the deletion cases
        // use, so the only difference between control and subject is the
        // tombstone.
        let probe = hnsw.entry().expect("non-empty index has an entry point");
        let budget = N;

        let res = knn(&hnsw, &features[probe], budget, budget);
        assert_eq!(
            res.len(),
            budget,
            "seed {seed}: control under-filled ({} < {budget}) with no deletions — \
             the zero-slack budget is not achievable, so this fixture cannot \
             isolate the tombstone bug",
            res.len()
        );

        let res_rt = knn_runtime(&rt, &features[probe], budget, budget);
        assert_eq!(
            res_rt.len(),
            budget,
            "seed {seed}: runtime control under-filled ({} < {budget}) with no \
             deletions",
            res_rt.len()
        );
    }
}

/// Delete the node the search seeds navigation from, then query at its old
/// position with a budget of exactly `live_count`. Every live node fits, so the
/// result set must come back completely full.
///
/// Fails today by exactly one on every trial: the tombstone is carried into
/// `nearest` by `initialize_searcher` / `lower_search`, holds a `cap` slot for
/// the whole zero-layer search, and is only scrubbed afterwards.
#[test]
fn deleted_seed_does_not_steal_a_result_slot() {
    let mut failures = Vec::new();

    for seed in 0..TRIALS {
        let features = corpus(seed);
        let mut hnsw = build(&features, seed);

        // On a freshly built index `entry()` returns the very node the search
        // seeds from: it scans the towers top-down and takes the first live
        // node, which is `layers.last()[0].zero_node` — exactly the node
        // `entry_feature()` hands to `initialize_searcher`.
        let seed_node = hnsw.entry().expect("non-empty index has an entry point");

        hnsw.mark_delete(seed_node);

        let live = hnsw.live_count();
        assert_eq!(live, N - 1, "seed {seed}: exactly one node was deleted");

        // Query at the deleted seed's old position so the descent carries it all
        // the way down: at distance zero nothing can displace it, so it reaches
        // the zero layer sitting in `nearest`.
        let res = knn(&hnsw, &features[seed_node], live, live);

        for n in &res {
            assert_ne!(
                n.index, seed_node,
                "seed {seed}: the deleted seed node leaked into results"
            );
        }

        if res.len() != live {
            failures.push(format!(
                "seed {seed}: got {} of {live} live neighbors",
                res.len()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "the deleted navigation seed consumed a result slot in {}/{TRIALS} trials \
         (the control fills completely at the same budget):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// The runtime-degree index mirrors the const-generic one line for line,
/// including the same post-search `retain`, so it carries the same defect and
/// must be fixed and pinned identically.
#[test]
fn runtime_deleted_seed_does_not_steal_a_result_slot() {
    let mut failures = Vec::new();

    for seed in 0..TRIALS {
        let features = corpus(seed);
        let mut hnsw = build_runtime(&features, seed);

        let seed_node = hnsw.entry().expect("non-empty index has an entry point");
        hnsw.mark_delete(seed_node);

        let live = hnsw.live_count();
        assert_eq!(live, N - 1, "seed {seed}: exactly one node was deleted");

        let res = knn_runtime(&hnsw, &features[seed_node], live, live);

        for n in &res {
            assert_ne!(
                n.index, seed_node,
                "seed {seed}: the deleted seed node leaked into runtime results"
            );
        }

        if res.len() != live {
            failures.push(format!(
                "seed {seed}: got {} of {live} live neighbors",
                res.len()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "the deleted navigation seed consumed a runtime result slot in \
         {}/{TRIALS} trials:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// The const-generic and runtime indexes must not diverge under the zero-slack
/// deleted-seed probe. Built from the same corpus with the same seed and
/// degrees, with the same node tombstoned, they must return the identical
/// ranked list.
#[test]
fn const_and_runtime_agree_on_deleted_seed_probe() {
    for seed in 0..TRIALS {
        let features = corpus(seed);
        let mut cst = build(&features, seed);
        let mut rt = build_runtime(&features, seed);

        let seed_node = cst.entry().expect("non-empty index has an entry point");
        assert_eq!(
            seed_node,
            rt.entry()
                .expect("non-empty runtime index has an entry point"),
            "seed {seed}: const and runtime must agree on the entry point"
        );

        cst.mark_delete(seed_node);
        rt.mark_delete(seed_node);

        let live = cst.live_count();
        assert_eq!(live, rt.live_count());

        assert_eq!(
            knn(&cst, &features[seed_node], live, live),
            knn_runtime(&rt, &features[seed_node], live, live),
            "seed {seed}: runtime diverged from const on the deleted-seed probe"
        );
    }
}
