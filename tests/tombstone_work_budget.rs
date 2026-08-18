//! Soft-deleted nodes must obey the search's work budget.
//!
//! ## The defect
//!
//! `search_single_layer` admits a neighbour to the traversal frontier only if it
//! falls within the beam — `pos = nearest.partition_point(..)` and both the heap
//! insert and the `candidates.push` happen inside `if pos != cap`.
//!
//! Soft-deleted nodes took an earlier branch that pushed onto `candidates`
//! UNCONDITIONALLY. Tombstones therefore escaped the bound that limits live
//! nodes, and the search expanded the entire tombstoned subgraph however distant
//! it was.
//!
//! ## Why this was not visible as "bad recall"
//!
//! It made search accidentally EXHAUSTIVE. Measured before the fix (N=4000,
//! dim=64, uniform corpus; work at ef=32, recall at ef=16):
//!
//! | density | evals/query | % of N | recall@10 |
//! | ---     | ---         | ---    | ---       |
//! | 25%     | 3973.1      |  99.3  | 1.0000    |
//! | 50%     | 4039.8      | 101.0  | 1.0000    |
//! | 75%     | 4048.4      | 101.2  | 1.0000    |
//! | 90%     | 4048.4      | 101.2  | 1.0000    |
//!
//! Recall was a perfect 1.0 at every density — because the index was scanning
//! everything. A linear scan with perfect recall is not an approximate index;
//! `ef` is the caller's work budget and it was being ignored.
//!
//! The tell is the FLATNESS: work barely moves between 25% and 90% deletion
//! (3973 -> 4048), and is already at 99.3% of N when only a quarter of the nodes
//! are deleted. Work that does not respond to the amount of data is not being
//! bounded at all.
//!
//! ## After the fix
//!
//! | density | evals/query | % of N | recall@10 | rebuilt-live recall |
//! | ---     | ---         | ---    | ---       | ---                 |
//! | 25%     | 2211.4      |  55.3  | 0.9283    | 0.9100              |
//! | 50%     | 2688.1      |  67.2  | 0.9483    | 0.9367              |
//! | 75%     | 3368.5      |  84.2  | 0.9850    | 0.9800              |
//! | 90%     | 3866.9      |  96.7  | 0.9983    | 0.9983              |
//!
//! Work now scales with density, and recall matches or slightly exceeds an index
//! REBUILT containing only the surviving vectors. That comparison is the one that
//! matters, and it is the strongest claim available: no material recall
//! regression against the behaviour soft-delete is supposed to be
//! indistinguishable from.
//!
//! It is NOT a claim that reachability is globally preserved. A tombstone
//! outside the beam is no longer expanded, so a live node reachable only through
//! a distant tombstone will not be found. That is deliberate approximate-search
//! pruning — an equally distant LIVE intermediate is pruned identically — and
//! the two hand-constructed tests in `hnsw_const.rs::beam_gate_tests` pin both
//! halves of that behaviour directly, which a statistical recall test is too
//! coarse to do.
//!
//! ## What is NOT fixed here
//!
//! Work is still 1.3x-9.4x the rebuilt-live baseline, because tombstoned nodes
//! are still stored, still linked, and still distance-evaluated when they fall
//! within the beam. Removing that residue requires compaction — physically
//! rebuilding the graph without the dead nodes — which is a separate change.
//! This test pins the bound, not the residue.

use hnsw::{Hnsw, HnswRuntime, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};
use std::cell::Cell;

thread_local! {
    /// Distance-evaluation counter.
    ///
    /// Thread-local, NOT a global atomic. Rust runs test functions on separate
    /// threads concurrently, and a shared counter is incremented by every test
    /// at once: the first version of this file used a `static AtomicUsize` and
    /// measured 212.6% of N for a search that actually performs 55.3%, because
    /// two tests were summing into it. The metric is only ever invoked on the
    /// thread driving the query, so a thread-local is both correct and cheaper.
    static EVALS: Cell<usize> = const { Cell::new(0) };
}

fn evals_reset() {
    EVALS.with(|e| e.set(0));
}

fn evals_get() -> usize {
    EVALS.with(|e| e.get())
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

const N: usize = 4000;
/// Dimension 64 is a measured choice, not a default. A probe across corpus
/// designs found dim=8 saturates recall at 1.0 for every `ef` (so a recall
/// assertion there is inert), while dim=64 moves recall 0.8050 -> 0.9900 across
/// `ef` 10..64 and can therefore actually register a regression. Clustered
/// corpora were low but FLAT (0.5350 -> 0.5383), measuring cluster reachability
/// rather than search quality, so they were rejected too.
const DIM: usize = 64;
const K: usize = 10;
/// `ef` for the work tests.
const EF: usize = 32;
/// `ef` for the RECALL test, deliberately lower.
///
/// At `ef = 32` the tombstoned recalls are 0.9883/0.9883/0.9950/1.0000 across
/// the four densities — close enough to the ceiling that a regression has little
/// room to show. At `ef = 16` they are 0.9283/0.9483/0.9850/0.9983, which leaves
/// headroom for a deficit to appear. The recall figures quoted in the module
/// documentation are the `ef = 16` ones.
const EF_RECALL: usize = 16;

type Idx = Hnsw<Counting, Vec<f32>, Pcg64, 12, 24>;

fn corpus(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect()
}

fn queries(count: usize) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(777);
    (0..count)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
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

/// Delete a deterministic fraction, returning the surviving ids.
fn delete_fraction(hnsw: &mut Idx, density: f64) -> Vec<usize> {
    let mut live = Vec::new();
    let step = 1.0 / density;
    let mut next = 0.0f64;
    for id in 0..N {
        if (id as f64) >= next {
            hnsw.mark_delete(id);
            next += step;
        } else {
            live.push(id);
        }
    }
    live
}

fn work_per_query(hnsw: &Idx, qs: &[Vec<f32>]) -> f64 {
    evals_reset();
    let mut searcher = Searcher::default();
    for q in qs {
        let mut dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            K
        ];
        hnsw.nearest(q, EF, &mut searcher, &mut dest);
    }
    evals_get() as f64 / qs.len() as f64
}

/// THE assertion for this defect.
///
/// Both ends of the ratio are stable, fixed baselines rather than a comparison
/// between two tombstoned configurations:
///
/// * `w_clean` — the same graph with nothing deleted. The floor.
/// * `w_full` — the same graph with EVERYTHING deleted. Every node is a
///   tombstone, so this is exactly the unbounded/full-scan work the defect
///   produced. The ceiling.
///
/// The assertion is then on the normalised excess over the clean baseline,
/// `(w_25 - w_clean) / (w_full - w_clean)`: what fraction of the full-scan
/// penalty survives. An earlier version compared work at 25% density against
/// work at 90% density, which fails the old implementation but could also be
/// satisfied by making the 90% case WORSE — its denominator was not a fixed
/// reference.
///
/// Distance-evaluation counts here are deterministic: the corpus, the PRNG and
/// the traversal are all seeded, so this is an exact contract and not a
/// performance heuristic sensitive to the host.
#[test]
fn tombstone_traversal_respects_the_work_budget() {
    /// Predeclared: at least half the full-scan excess must be eliminated.
    /// Measured at roughly 0.15 after the fix and 0.97 before it, so the bound
    /// sits between the two regimes with wide margin on both sides.
    const MAX_EXCESS_FRACTION: f64 = 0.5;

    let features = corpus(11);
    let qs = queries(40);

    let clean = build(&features);
    let w_clean = work_per_query(&clean, &qs);

    let mut all_dead = build(&features);
    for id in 0..N {
        all_dead.mark_delete(id);
    }
    let w_full = work_per_query(&all_dead, &qs);

    let mut sparse = build(&features);
    let _ = delete_fraction(&mut sparse, 0.25);
    let w_25 = work_per_query(&sparse, &qs);

    assert!(
        w_full > w_clean,
        "baseline sanity: a fully tombstoned index ({w_full:.1}) must cost more \
         than a clean one ({w_clean:.1}), or the fixture cannot measure anything"
    );

    let excess = (w_25 - w_clean) / (w_full - w_clean);
    assert!(
        excess < MAX_EXCESS_FRACTION,
        "at 25% deletion density the search retained {:.0}% of the full-scan \
         penalty ({w_25:.1} evals/query against {w_clean:.1} clean and \
         {w_full:.1} fully tombstoned). Tombstones are bypassing the beam bound, \
         so the search expands the whole tombstoned subgraph regardless of \
         distance — a linear scan wearing an index's clothes.",
        excess * 100.0
    );
}

/// The bound must not be bought with reachability. At high density most paths
/// between live nodes run THROUGH tombstones, so gating them out of the frontier
/// could disconnect live regions.
///
/// The reference is an index REBUILT with only the surviving vectors — the
/// behaviour a soft-deleted index is supposed to be indistinguishable from.
/// Measured parity: 0.9283 vs 0.9100 at 25%, 0.9983 vs 0.9983 at 90%.
#[test]
fn recall_matches_an_index_rebuilt_without_the_deleted_nodes() {
    /// Predeclared: the tombstoned index may not trail the rebuilt one by more
    /// than this. Measurement shows it currently LEADS at every density, so any
    /// deficit at all indicates lost reachability.
    const MAX_DEFICIT: f64 = 0.02;

    let features = corpus(11);
    let qs = queries(60);

    for density in [0.25f64, 0.90] {
        let mut tombstoned = build(&features);
        let live = delete_fraction(&mut tombstoned, density);
        let live_feats: Vec<Vec<f32>> = live.iter().map(|&i| features[i].clone()).collect();
        let rebuilt = build(&live_feats);

        let mut tomb_hits = 0usize;
        let mut rebuilt_hits = 0usize;
        let mut total = 0usize;

        for q in &qs {
            // Truth over the LIVE set only.
            let mut all: Vec<(u64, usize)> = live_feats
                .iter()
                .enumerate()
                .map(|(i, f)| (Counting.distance(q, f), i))
                .collect();
            all.sort();
            let truth_positions: Vec<usize> = all.into_iter().take(K).map(|(_, i)| i).collect();
            // Same points, expressed as ids in the tombstoned index.
            let truth_original: std::collections::BTreeSet<usize> =
                truth_positions.iter().map(|&p| live[p]).collect();
            let truth_rebuilt: std::collections::BTreeSet<usize> =
                truth_positions.into_iter().collect();

            let mut s = Searcher::default();
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            let got = tombstoned.nearest(q, EF_RECALL, &mut s, &mut dest);
            tomb_hits += got
                .iter()
                .filter(|n| truth_original.contains(&n.index))
                .count();

            let mut s2 = Searcher::default();
            let mut dest2 = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            let got2 = rebuilt.nearest(q, EF_RECALL, &mut s2, &mut dest2);
            rebuilt_hits += got2
                .iter()
                .filter(|n| truth_rebuilt.contains(&n.index))
                .count();

            total += K;
        }

        let tomb = tomb_hits as f64 / total as f64;
        let reference = rebuilt_hits as f64 / total as f64;
        assert!(
            tomb >= reference - MAX_DEFICIT,
            "at {:.0}% deletion density the tombstoned index recalled {tomb:.4} \
             against {reference:.4} for an index rebuilt without the deleted \
             nodes — a deficit of {:.4}, over the {MAX_DEFICIT} budget. The beam \
             gate has cost reachability: live nodes reachable only through a \
             distant tombstone can no longer be found.",
            density * 100.0,
            reference - tomb
        );
    }
}

/// A deleted node must never surface, and results must not under-fill while live
/// nodes remain. Under-filling is how a reachability failure shows up before it
/// is visible as poor recall.
#[test]
fn results_stay_full_and_free_of_tombstones() {
    let features = corpus(11);
    let qs = queries(40);

    for density in [0.75f64, 0.90, 0.95] {
        let mut hnsw = build(&features);
        let live = delete_fraction(&mut hnsw, density);
        let live_set: std::collections::BTreeSet<usize> = live.iter().copied().collect();
        let want = K.min(live.len());

        let mut searcher = Searcher::default();
        for q in &qs {
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            let got = hnsw.nearest(q, EF, &mut searcher, &mut dest);

            for n in got.iter() {
                assert!(
                    live_set.contains(&n.index),
                    "a deleted node ({}) was returned at {:.0}% density",
                    n.index,
                    density * 100.0
                );
            }
            assert_eq!(
                got.len(),
                want,
                "returned {} of {want} results at {:.0}% density with {} live \
                 nodes remaining",
                got.len(),
                density * 100.0,
                live.len()
            );
        }
    }
}

/// The runtime-degree index carries its own copy of `search_single_layer`, so
/// the same gate has to be verified there or half the change is unasserted.
#[test]
fn runtime_index_also_respects_the_work_budget() {
    fn build_rt(features: &[Vec<f32>]) -> HnswRuntime<Counting, Vec<f32>, Pcg64> {
        let mut searcher = Searcher::default();
        let mut hnsw = HnswRuntime::new(Counting, 12, 24);
        for f in features {
            hnsw.insert(f.clone(), &mut searcher);
        }
        hnsw
    }

    let features = corpus(11);
    let qs = queries(40);

    let measure = |hnsw: &HnswRuntime<Counting, Vec<f32>, Pcg64>| -> f64 {
        evals_reset();
        let mut searcher = Searcher::default();
        for q in &qs {
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            hnsw.nearest(q, EF, &mut searcher, &mut dest);
        }
        evals_get() as f64 / qs.len() as f64
    };

    let mut sparse = build_rt(&features);
    let mut dense = build_rt(&features);
    let step_delete = |h: &mut HnswRuntime<Counting, Vec<f32>, Pcg64>, density: f64| {
        let step = 1.0 / density;
        let mut next = 0.0f64;
        for id in 0..N {
            if (id as f64) >= next {
                h.mark_delete(id);
                next += step;
            }
        }
    };
    step_delete(&mut sparse, 0.25);
    step_delete(&mut dense, 0.90);

    let w_sparse = measure(&sparse);
    let w_dense = measure(&dense);
    let ratio = w_sparse / w_dense;

    assert!(
        ratio < 0.85,
        "runtime index: work barely responds to deletion density ({w_sparse:.1} \
         vs {w_dense:.1}, ratio {ratio:.2}) — tombstones are bypassing the beam \
         bound on this index"
    );
}
