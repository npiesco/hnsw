//! In-traversal filtered search: a caller predicate must be applied DURING graph
//! traversal, not as a post-filter over an over-fetched result set.
//!
//! ## What this pins
//!
//! The consumer pattern this replaces is "over-fetch by a constant, then retain":
//!
//! ```ignore
//! let mut results = self.search(query, k * 10)?;
//! results.retain(|(id, _)| filter.evaluate(*id));
//! results.truncate(k);
//! ```
//!
//! With a selective predicate that silently returns fewer than `k`, with no
//! signal to the caller, because the matching nodes simply are not in the top
//! `k * 10` unfiltered. Widening the constant (`k * 20`, ...) only moves the
//! selectivity at which it breaks.
//!
//! `nearest_filtered` instead admits only passing nodes to the bounded result
//! heap, while rejected nodes remain eligible as NAVIGATION INTERMEDIATES so
//! accepted nodes behind them can still be reached — eligible, not guaranteed,
//! since they must still fall within the beam and the early stop may end the
//! search first. This is the same shape as the soft-delete tombstone guard.
//!
//! ## Budget semantics (deliberate, and asserted below)
//!
//! The filtered zero-layer frontier is a binary min-heap expanded nearest-first,
//! with hnswlib's early stop: while a full set of `ef` accepted results is held
//! and the closest remaining candidate is farther than the worst of them, the
//! search terminates. That is the standard HNSW termination heuristic, not a
//! proof — a farther graph node can lead to a nearer neighbour, which is exactly
//! why the search is approximate.
//!
//! `filtered_search_does_not_degenerate_to_a_full_scan` is what holds that down.
//! Without it, "returns a full k" would be satisfied by an implementation that
//! simply visited every node in the graph, which is exactly what an earlier
//! revision of this feature did.
//!
//! Note the bound comes from the traversal ORDER and the early stop, not from
//! the admission rule. The rejected-node beam check is retained because it keeps
//! out-of-beam rejected nodes out of the frontier, bounding its SIZE — measured
//! at up to 3.4x fewer pushes and a 3.5x smaller peak frontier, with
//! byte-identical distance-evaluation counts, so a work-count test cannot
//! observe it. The one exception is an exact distance TIE, where the gate does
//! change evaluations and results because the early stop tests a strict `>`
//! while the beam check tests `>=`; ties are negligible on a continuous corpus
//! but ordinary for a discrete metric. See `filtered_tie_tests` in
//! `hnsw_const.rs`.

//!
//! This is a distance/beam bound, NOT a hard work bound — see the
//! `nearest_filtered` rustdoc.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};

use hnsw::{Hnsw, HnswRuntime, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

/// Euclidean distance that counts how many times it is evaluated, so a test can
/// assert on the actual work performed rather than only on the results.
struct CountingEuclidean {
    calls: AtomicUsize,
}

impl CountingEuclidean {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }
    fn take(&self) -> usize {
        self.calls.swap(0, Ordering::Relaxed)
    }
}

impl Metric<Vec<f32>> for &CountingEuclidean {
    type Unit = u64;
    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> u64 {
        self.calls.fetch_add(1, Ordering::Relaxed);
        a.iter()
            .zip(b.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
            .to_bits() as u64
    }
}

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
const N: usize = 4000;
const M: usize = 12;
const M0: usize = 24;
const K: usize = 10;
const EF: usize = 32;
/// 1 in 20 nodes match, i.e. 5% selectivity — low enough that a `k * 10`
/// over-fetch cannot reliably find `k` of them.
const SELECTIVITY_STEP: usize = 20;

fn matches(id: usize) -> bool {
    id.is_multiple_of(SELECTIVITY_STEP)
}

fn corpus(seed: u64) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(seed);
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

fn dest(k: usize) -> Vec<Neighbor<u64>> {
    vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ]
}

/// Exact top-k over nodes accepted by `pred`.
fn ground_truth(
    features: &[Vec<f32>],
    q: &[f32],
    k: usize,
    pred: impl Fn(usize) -> bool,
) -> Vec<usize> {
    let qv = q.to_vec();
    let mut all: Vec<(u64, usize)> = (0..features.len())
        .filter(|&i| pred(i))
        .map(|i| (Euclidean.distance(&qv, &features[i]), i))
        .collect();
    all.sort();
    all.into_iter().take(k).map(|(_, i)| i).collect()
}

/// Faithful emulation of the consumer's current strategy: over-fetch by a
/// constant factor, then drop non-matching results.
fn post_filter_emulation(
    hnsw: &Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0>,
    q: &[f32],
    k: usize,
    over_fetch: usize,
) -> Vec<Neighbor<u64>> {
    let wide = k * over_fetch;
    let mut searcher = Searcher::default();
    let mut d = dest(wide);
    let mut got = hnsw
        .nearest(&q.to_vec(), wide.max(EF), &mut searcher, &mut d)
        .to_vec();
    got.retain(|n| matches(n.index));
    got.truncate(k);
    got
}

fn queries(features: &[Vec<f32>], count: usize) -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(0xC0FFEE);
    (0..count)
        .map(|_| {
            let base = rng.gen_range(0..N);
            features[base]
                .iter()
                .map(|&x| x + rng.gen_range(-0.1f32..0.1f32))
                .collect()
        })
        .collect()
}

/// The headline behavior: with a 5%-selective predicate, in-traversal filtering
/// returns a full `k` of genuinely matching neighbors, while the over-fetch
/// strategy the consumer uses today silently under-returns.
///
/// Both halves matter. Asserting only that the new API returns `k` would be
/// satisfied by an implementation identical to the old one on an easier fixture;
/// the differential against `post_filter_emulation` on the SAME queries is what
/// proves the new path is actually better.
#[test]
fn filtered_search_fills_k_where_post_filtering_under_returns() {
    let features = corpus(1);
    let hnsw = build(&features);
    let qs = queries(&features, 100);

    let mut filtered_short = 0usize;
    let mut post_short = 0usize;
    let mut filtered_hits = 0usize;
    let mut truth_total = 0usize;

    for q in &qs {
        let truth: BTreeSet<usize> = ground_truth(&features, q, K, matches).into_iter().collect();
        assert_eq!(
            truth.len(),
            K,
            "fixture must have at least K matching nodes"
        );

        let mut searcher = Searcher::default();
        let mut d = dest(K);
        let got = hnsw
            .nearest_filtered(&q.to_vec(), EF, &mut searcher, &mut d, &matches)
            .to_vec();

        for n in &got {
            assert!(
                matches(n.index),
                "non-matching node {} was returned by nearest_filtered",
                n.index
            );
        }
        if got.len() < K {
            filtered_short += 1;
        }
        filtered_hits += got.iter().filter(|n| truth.contains(&n.index)).count();
        truth_total += truth.len();

        if post_filter_emulation(&hnsw, q, K, 10).len() < K {
            post_short += 1;
        }
    }

    let recall = filtered_hits as f64 / truth_total as f64;

    assert_eq!(
        filtered_short,
        0,
        "nearest_filtered under-returned on {filtered_short}/{} queries",
        qs.len()
    );
    assert!(
        recall >= 0.90,
        "filtered recall@{} was {:.4}, below the 0.90 floor — returning a \
         full k of the WRONG matching nodes is not a fix",
        K,
        recall
    );
    assert!(
        post_short > qs.len() / 2,
        "the post-filter emulation was expected to under-return on most queries \
         at 5% selectivity, but only {post_short}/{} were short — the fixture is \
         not actually exercising the defect this API exists to fix",
        qs.len()
    );
}

/// The result heap is bounded, but the frontier must be too — otherwise a
/// selective predicate degenerates into a full scan and "returns a full k"
/// becomes a meaningless assertion.
///
/// ## What this guards, and how the threshold was chosen
///
/// The fixture is fully seeded and deterministic, so these figures are exactly
/// reproducible. At `ef = 32` and 5% selectivity, distance evaluations per query
/// against `N = 4000`, both measured on this tree by ablation:
///
/// | zero-layer frontier                    | evals/query | % of N |
/// |----------------------------------------|------------:|-------:|
/// | best-first min-heap + early stop       |        2117 |  52.9% |
/// | same, early stop disabled              |        2453 |  61.3% |
///
/// The 0.58 threshold sits between them, and was verified to go red when the
/// early stop is removed. An earlier revision using a depth-first frontier
/// measured over 100% of N — worse than a brute-force scan — so this assertion
/// is guarding against a regression that genuinely happened, not a hypothetical.
///
/// This is FIXTURE-SPECIFIC and calibrated from measurement, not an API
/// invariant. Work still scales roughly as `ef / selectivity`, which is inherent
/// to filtered ANN.
#[test]
fn filtered_search_does_not_degenerate_to_a_full_scan() {
    let features = corpus(1);
    let metric = CountingEuclidean::new();

    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<&CountingEuclidean, Vec<f32>, Pcg64, M, M0> =
        Hnsw::new_params_and_prng(&metric, Default::default(), Pcg64::seed_from_u64(42));
    for f in &features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    metric.take();

    let qs = queries(&features, 20);
    let mut total = 0usize;
    let predicate_calls = AtomicUsize::new(0);

    for q in &qs {
        let counted = |id: usize| {
            predicate_calls.fetch_add(1, Ordering::Relaxed);
            matches(id)
        };
        let mut s = Searcher::default();
        let mut d = dest(K);
        let _ = hnsw.nearest_filtered(&q.to_vec(), EF, &mut s, &mut d, &counted);
        total += metric.take();
    }

    let per_query = total as f64 / qs.len() as f64;
    println!(
        "filtered search: {per_query:.0} distance evals/query out of {N} nodes \
         ({:.1}%), {} predicate calls total",
        100.0 * per_query / N as f64,
        predicate_calls.load(Ordering::Relaxed)
    );

    assert!(
        per_query < N as f64 * 0.58,
        "filtered search performed {per_query:.0} distance evaluations per query \
         against a corpus of {N} ({:.1}%), over the 58% budget. Measured by \
         ablation on this exact fixture: 52.9% with the best-first min-heap \
         frontier and its early stop, 61.3% with the early stop removed. A \
         number at or above that means the filtered zero-layer traversal lost \
         its ordering or its termination check.",
        100.0 * per_query / N as f64
    );
}

/// A predicate that accepts everything must return EXACTLY what no predicate
/// returns.
///
/// This asserts full equality of the ranked result lists, which became a real
/// contract rather than an accident once the unfiltered zero-layer query moved
/// to best-first. Both paths now run the same sequence: `initialize_searcher`,
/// an unfiltered descent through the upper layers at `cap = 1`, the tombstone
/// scrub, then `search_zero_layer_best_first` with the same `ef` — one passing
/// the caller's predicate, the other an accept-all closure. With an accept-all
/// predicate those are the same computation, so any divergence means the two
/// entry points have drifted apart and one of them is doing something the other
/// is not.
///
/// An earlier revision deliberately did NOT assert this, on the grounds that
/// "the filtered path uses a best-first frontier with an early stop while the
/// unfiltered path is depth-first, so they are different algorithms and any
/// byte-for-byte agreement is incidental to the fixture". That was true when
/// written. It stopped being true when queries moved to best-first, and the
/// weaker `recall > 0.5` check it left behind would pass even if the filtered
/// path silently lost half its results.
///
/// What is still NOT a contract: recall parity at equal `ef` between this
/// fixture and some other traversal. An earlier revision asserted
/// `all_r >= plain_r - 0.01`, which was FALSE and passed only because this
/// fixture's `DIM = 8` saturates recall at 1.0 for both paths. Measured on a
/// non-saturating corpus (dim=64, N=4000, k=10) against the OLD depth-first
/// unfiltered path:
///
/// | ef | plain evals | plain recall | filtered evals | filtered recall |
/// | -- | ----------- | ------------ | -------------- | --------------- |
/// | 16 | 1190.4      | 0.8650       | 450.7          | 0.6267          |
/// | 32 | 1894.7      | 0.9617       | 707.3          | 0.7933          |
/// | 64 | 2669.7      | 0.9933       | 1143.7         | 0.9217          |
///
/// The best-first path recalled materially worse at every equal `ef` — by 0.17
/// at ef=32, seventeen times the old 0.01 budget — because `ef` is result-list
/// capacity rather than an expansion budget, and the depth-first path bought its
/// extra recall with roughly 2.6x the work. That measurement is what moved the
/// unfiltered path onto best-first, and it is retained here because it explains
/// why a same-`ef` recall comparison between DIFFERENT traversals is not a
/// contract even though an equality comparison between the SAME traversal is.
#[test]
fn accept_all_predicate_matches_unfiltered_search() {
    for seed in [2u64, 5, 11] {
        let features = corpus(seed);
        let hnsw = build(&features);
        let rt = build_runtime(&features);

        let mut plain_hits = 0usize;
        let mut all_hits = 0usize;
        let mut rt_plain_hits = 0usize;
        let mut rt_all_hits = 0usize;
        let mut total = 0usize;

        for q in queries(&features, 50) {
            let truth: BTreeSet<usize> = ground_truth(&features, &q, K, |_| true)
                .into_iter()
                .collect();

            let mut s1 = Searcher::default();
            let mut d1 = dest(K);
            let plain = hnsw.nearest(&q, EF, &mut s1, &mut d1).to_vec();

            let mut s2 = Searcher::default();
            let mut d2 = dest(K);
            let all = hnsw
                .nearest_filtered(&q, EF, &mut s2, &mut d2, &|_| true)
                .to_vec();

            assert_eq!(all.len(), K, "accept-all filtered search under-filled");
            assert_eq!(
                all, plain,
                "seed {seed}: an accept-all predicate must return exactly what \
                 unfiltered search returns — both entry points run the same \
                 descent and the same `search_zero_layer_best_first` at the same \
                 `ef`, so a divergence means they have drifted apart"
            );

            let mut s3 = Searcher::default();
            let mut d3 = dest(K);
            let rt_plain = rt.nearest(&q, EF, &mut s3, &mut d3).to_vec();

            let mut s4 = Searcher::default();
            let mut d4 = dest(K);
            let rt_all = rt
                .nearest_filtered(&q, EF, &mut s4, &mut d4, &|_| true)
                .to_vec();

            assert_eq!(
                rt_all.len(),
                K,
                "runtime accept-all filtered search under-filled"
            );
            assert_eq!(
                rt_all, rt_plain,
                "seed {seed}: the runtime index must show the same accept-all \
                 equality as the const-generic one"
            );

            plain_hits += plain.iter().filter(|n| truth.contains(&n.index)).count();
            all_hits += all.iter().filter(|n| truth.contains(&n.index)).count();
            rt_plain_hits += rt_plain.iter().filter(|n| truth.contains(&n.index)).count();
            rt_all_hits += rt_all.iter().filter(|n| truth.contains(&n.index)).count();
            total += truth.len();
        }

        let (plain_r, all_r) = (
            plain_hits as f64 / total as f64,
            all_hits as f64 / total as f64,
        );
        let (rt_plain_r, rt_all_r) = (
            rt_plain_hits as f64 / total as f64,
            rt_all_hits as f64 / total as f64,
        );

        // Recall is recorded for both paths but NOT compared at equal `ef`
        // against a different traversal; see the doc comment. The contract that
        // does hold is exact equality between the two entry points, asserted
        // per-query above.
        assert!(
            all_r > 0.5,
            "seed {}: accept-all filtered recall collapsed to {:.4}; the predicate \
             accepts everything, so this should behave like a normal search",
            seed,
            all_r
        );
        assert!(
            rt_all_r > 0.5,
            "seed {}: runtime accept-all filtered recall collapsed to {:.4}",
            seed,
            rt_all_r
        );
        // Sanity that the fixture is not degenerate for the plain path either.
        assert!(
            plain_r > 0.5 && rt_plain_r > 0.5,
            "seed {}: unfiltered recall is {:.4}/{:.4}; the fixture is broken",
            seed,
            plain_r,
            rt_plain_r
        );
    }
}

/// The navigation seed is pushed straight into `nearest` by
/// `initialize_searcher`, and re-pushed at every descent by `lower_search`,
/// without consulting the predicate. If it is not scrubbed before the zero-layer
/// search it occupies a result slot — the same defect soft-delete had.
///
/// This constructs the case explicitly rather than hoping the selective fixture
/// happens to reject the seed.
#[test]
fn a_rejected_navigation_seed_does_not_occupy_a_result_slot() {
    let features = corpus(3);
    let hnsw = build(&features);

    let seed_node = hnsw.entry().expect("non-empty index has an entry point");

    // Reject ONLY the seed. Every other node matches, so a correct
    // implementation must return a completely full k of live, accepted nodes.
    let pred = |id: usize| id != seed_node;

    for q in queries(&features, 50) {
        let truth: BTreeSet<usize> = ground_truth(&features, &q, K, pred).into_iter().collect();

        let mut s = Searcher::default();
        let mut d = dest(K);
        let got = hnsw
            .nearest_filtered(&q, EF, &mut s, &mut d, &pred)
            .to_vec();

        assert_eq!(
            got.len(),
            K,
            "result set under-filled ({} < {K}) when the navigation seed was \
             rejected — the seed consumed a result slot",
            got.len()
        );
        for n in &got {
            assert_ne!(
                n.index, seed_node,
                "the rejected navigation seed leaked into results"
            );
        }
        assert!(
            got.iter().filter(|n| truth.contains(&n.index)).count() * 4 >= K * 3,
            "recall collapsed when the seed was rejected"
        );
    }

    // And the degenerate case: query AT the seed's own position, where it is at
    // distance zero and nothing can displace it.
    let mut s = Searcher::default();
    let mut d = dest(K);
    let got = hnsw
        .nearest_filtered(&features[seed_node], EF, &mut s, &mut d, &pred)
        .to_vec();
    assert_eq!(
        got.len(),
        K,
        "querying at the rejected seed's own position under-filled"
    );
    assert!(got.iter().all(|n| n.index != seed_node));
}

/// Tombstones and caller predicates are independent exclusion mechanisms and
/// must compose: neither a deleted node nor a rejected node may surface, and
/// together they must still not consume result slots.
#[test]
fn predicate_and_tombstones_compose() {
    let features = corpus(4);
    let mut hnsw = build(&features);

    // Delete every 7th node — deliberately overlapping the `% 20` predicate on
    // multiples of 140, so both mechanisms exclude some of the same nodes.
    let deleted: BTreeSet<usize> = (0..N).step_by(7).collect();
    for &d in &deleted {
        hnsw.mark_delete(d);
    }

    let live_and_matching = |id: usize| matches(id) && !deleted.contains(&id);

    for q in queries(&features, 50) {
        let truth: BTreeSet<usize> = ground_truth(&features, &q, K, live_and_matching)
            .into_iter()
            .collect();
        assert_eq!(truth.len(), K, "fixture must retain K live matching nodes");

        let mut s = Searcher::default();
        let mut d = dest(K);
        let got = hnsw
            .nearest_filtered(&q, EF, &mut s, &mut d, &matches)
            .to_vec();

        for n in &got {
            assert!(
                matches(n.index),
                "non-matching node {} surfaced under composition",
                n.index
            );
            assert!(
                !deleted.contains(&n.index),
                "soft-deleted node {} surfaced under composition",
                n.index
            );
        }
        assert_eq!(
            got.len(),
            K,
            "composition under-filled ({} < {K}) — a tombstone or a rejected node \
             consumed a result slot",
            got.len()
        );
    }
}

/// The runtime-degree index mirrors the const-generic one and must carry the
/// identical filtered-search guarantees.
#[test]
fn runtime_filtered_search_fills_k() {
    let features = corpus(1);
    let rt = build_runtime(&features);

    let mut short = 0usize;
    let mut hits = 0usize;
    let mut total = 0usize;

    let qs = queries(&features, 50);
    for q in &qs {
        let truth: BTreeSet<usize> = ground_truth(&features, q, K, matches).into_iter().collect();

        let mut s = Searcher::default();
        let mut d = dest(K);
        let got = rt
            .nearest_filtered(q, EF, &mut s, &mut d, &matches)
            .to_vec();

        for n in &got {
            assert!(
                matches(n.index),
                "runtime returned non-matching {}",
                n.index
            );
        }
        if got.len() < K {
            short += 1;
        }
        hits += got.iter().filter(|n| truth.contains(&n.index)).count();
        total += truth.len();
    }

    assert_eq!(
        short, 0,
        "runtime nearest_filtered under-returned on {short} queries"
    );
    let recall = hits as f64 / total as f64;
    assert!(
        recall >= 0.90,
        "runtime filtered recall@{} was {:.4}, below the 0.90 floor",
        K,
        recall
    );
}

/// An `ef` of 0 is a zero result budget and must return nothing.
///
/// This is not hypothetical pedantry. Before the guard, the seed could survive
/// the pre-search scrub and leave one element in `nearest`; the `len() == ef`
/// checks then never held, so neither the early stop nor the beam bound ever
/// engaged, and the search both returned a result against a zero budget and let
/// `nearest` grow without limit.
#[test]
fn zero_ef_returns_nothing_in_both_indexes() {
    let features = corpus(1);
    let hnsw = build(&features);
    let rt = build_runtime(&features);

    // Query AT an indexed point, which is the case most likely to smuggle a
    // result out: the seed and the query can coincide at distance zero.
    for probe in [0usize, 20, 1000] {
        let q = &features[probe];

        let mut s = Searcher::default();
        let mut d = dest(K);
        let got = hnsw.nearest_filtered(q, 0, &mut s, &mut d, &|_| true);
        assert!(
            got.is_empty(),
            "ef=0 returned {} results from the const index",
            got.len()
        );

        let mut s2 = Searcher::default();
        let mut d2 = dest(K);
        let got_rt = rt.nearest_filtered(q, 0, &mut s2, &mut d2, &|_| true);
        assert!(
            got_rt.is_empty(),
            "ef=0 returned {} results from the runtime index",
            got_rt.len()
        );
    }
}

/// The runtime index needs its own work bound. Its early stop can regress
/// without `runtime_filtered_search_fills_k` noticing, because that test only
/// checks results — a full scan returns perfectly good results.
#[test]
fn runtime_filtered_search_does_not_degenerate_to_a_full_scan() {
    let features = corpus(1);
    let metric = CountingEuclidean::new();

    let mut searcher = Searcher::default();
    let mut rt: HnswRuntime<&CountingEuclidean, Vec<f32>, Pcg64> = HnswRuntime::new_params_and_prng(
        &metric,
        M,
        M0,
        Default::default(),
        Pcg64::seed_from_u64(42),
    );
    for f in &features {
        rt.insert(f.clone(), &mut searcher);
    }
    metric.take();

    let qs = queries(&features, 20);
    let mut total = 0usize;
    for q in &qs {
        let mut s = Searcher::default();
        let mut d = dest(K);
        let _ = rt.nearest_filtered(&q.to_vec(), EF, &mut s, &mut d, &matches);
        total += metric.take();
    }

    let per_query = total as f64 / qs.len() as f64;
    println!(
        "runtime filtered search: {per_query:.0} distance evals/query out of {N} \
         nodes ({:.1}%)",
        100.0 * per_query / N as f64
    );

    assert!(
        per_query < N as f64 * 0.58,
        "runtime filtered search performed {per_query:.0} distance evaluations \
         per query against a corpus of {N} ({:.1}%), over the 58% budget — the \
         runtime filtered traversal lost its ordering or its termination check. \
         Both indexes measure identically (2117 evals, 52.9%) on this fixture, \
         so a divergence here also means the two implementations drifted apart.",
        100.0 * per_query / N as f64
    );
}
