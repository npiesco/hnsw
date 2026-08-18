//! Filtered search must stay correct as matches become rare, and must never
//! cost more than the brute-force scan it degenerates into.
//!
//! ## The regime
//!
//! `search_zero_layer_best_first`'s early stop is guarded by
//! `nearest.len() == ef`. When fewer than `ef` nodes match the predicate that
//! condition never holds, so the search cannot terminate early and traverses the
//! whole graph.
//!
//! Measured (N=4000, dim=64, ef=32, k=10, 30 queries):
//!
//! | matches | evals/query | % of N | returned / available |
//! | ---     | ---         | ---    | ---                  |
//! | 4000    |   711.6     |  17.8  | 300 / 300            |
//! | 2000    |  1105.4     |  27.6  | 300 / 300            |
//! | 1000    |  1763.7     |  44.1  | 300 / 300            |
//! |  400    |  2751.8     |  68.8  | 300 / 300            |
//! |   80    |  3987.5     |  99.7  | 300 / 300            |
//! |   20    |  4048.4     | 101.2  | 300 / 300            |
//! |    5    |  4048.4     | 101.2  | 150 / 150            |
//!
//! ## Why this is not treated as a defect
//!
//! The predicate is opaque. Nothing in the index knows where matching nodes are,
//! so the only way to find them is to reach them by traversing edges. When five
//! nodes out of four thousand match, any correct algorithm over this structure
//! must keep exploring until it has found them, and "keep exploring" over a graph
//! with no predicate index means visiting substantially everything. Stopping
//! earlier would silently drop matches that exist — trading a performance
//! property for a correctness one, which is the wrong trade.
//!
//! So the guarantee worth pinning is not sublinearity, which is unobtainable
//! here. It is that the degeneration is GRACEFUL:
//!
//! * every available match is still returned, at every selectivity; and
//! * the cost converges on a brute-force scan and does not exceed it.
//!
//! Both are asserted below. The second matters because the naive formulation of
//! filtered search — hnswlib's admission rule alone, without the best-first
//! frontier this crate needed — was measured visiting over 100% of the graph at
//! 2% selectivity, i.e. strictly worse than not using an index at all.

use hnsw::{Hnsw, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};
use std::cell::Cell;

thread_local! {
    /// Thread-local, not a global atomic: cargo runs test functions
    /// concurrently and a shared counter is summed across them.
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

const N: usize = 4000;
const DIM: usize = 64;
const K: usize = 10;
const EF: usize = 32;

type Idx = Hnsw<Counting, Vec<f32>, Pcg64, 12, 24>;

fn corpus() -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(11);
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

/// Rarity must cost time, not results — on this fixture.
///
/// Runs down to five matching nodes in four thousand, far past the point where
/// the early stop can fire, and requires every available match back. A search
/// that bailed out to bound its work would fail here, which is the tradeoff this
/// forbids.
///
/// Scope, stated precisely: this is graceful degradation ON THIS FIXTURE, not a
/// general completeness guarantee. When matches are fewer than `ef` the search
/// traverses the whole REACHABLE component, but HNSW topology does not guarantee
/// every matching node is reachable from the entry point — and `nearest_filtered`
/// documents that it may under-fill. A corpus whose matches sat in a component
/// the descent cannot enter would legitimately return fewer, so this asserts the
/// behaviour observed here rather than a property of the algorithm.
#[test]
fn every_available_match_is_returned_on_this_fixture() {
    let features = corpus();
    let hnsw = build(&features);
    let qs = queries(30);
    let mut searcher = Searcher::default();

    for every in [1usize, 10, 50, 200, 800] {
        let matches = N / every;
        let want = K.min(matches);
        let filter = move |id: usize| id.is_multiple_of(every);

        for (qi, q) in qs.iter().enumerate() {
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            let got = hnsw.nearest_filtered(q, EF, &mut searcher, &mut dest, &filter);

            assert_eq!(
                got.len(),
                want,
                "query {qi} with {matches} matching nodes in the index returned \
                 {} results, expected {want}. Filtered search must not trade \
                 completeness for a work bound.",
                got.len()
            );
            for n in got.iter() {
                assert!(
                    filter(n.index),
                    "query {qi} returned node {} which does not satisfy the \
                     predicate",
                    n.index
                );
            }
        }
    }
}

/// The degeneration must converge on a graph scan, not exceed it.
///
/// The right reference here is the number of NODES the traversal may examine,
/// not the cost of an optimal brute-force filtered search. Those are different
/// baselines and conflating them would overstate what this asserts: a
/// filter-first linear scan evaluates the predicate `N` times but the metric
/// only `matches` times — five metric calls for five matches, against this
/// search's 4048. So an ANN traversal is genuinely far more expensive than
/// optimal brute force in this regime, and this test does not claim otherwise.
///
/// What it does claim is that the traversal visits each node about once rather
/// than thrashing: the naive filtered formulation (hnswlib's admission rule
/// without the best-first frontier this crate needed) was measured visiting over
/// 100% of the graph at 2% selectivity — nodes re-entering the frontier and
/// being re-expanded. That is the regression this guards.
#[test]
fn a_rare_filter_visits_each_node_about_once() {
    /// Distance evaluations may reach N — the search genuinely has to look
    /// everywhere — plus a small allowance for the hierarchy descent, which
    /// re-evaluates a handful of upper-layer nodes. Measured at 4048.4 for
    /// N=4000, i.e. 1.012x; anything approaching 1.5x would mean nodes are being
    /// evaluated repeatedly rather than once.
    const MAX_RATIO: f64 = 1.10;

    let features = corpus();
    let hnsw = build(&features);
    let qs = queries(30);
    let mut searcher = Searcher::default();

    // 5 matches in 4000: deep in the regime where the early stop cannot fire.
    let filter = |id: usize| id.is_multiple_of(800);

    EVALS.with(|e| e.set(0));
    for q in &qs {
        let mut dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            K
        ];
        hnsw.nearest_filtered(q, EF, &mut searcher, &mut dest, &filter);
    }
    let per_query = EVALS.with(|e| e.get()) as f64 / qs.len() as f64;
    let ratio = per_query / N as f64;

    assert!(
        ratio <= MAX_RATIO,
        "an extremely selective filter cost {per_query:.1} distance evaluations \
         per query against {N} nodes ({ratio:.3}x). Filtered search may traverse \
         the whole reachable graph when matches are rarer than `ef`, but it must \
         visit each node about once — exceeding N means nodes are re-entering the \
         frontier and being re-expanded, which is what the pre-best-first \
         formulation did."
    );
}

/// Selectivity must actually drive the cost. If work were flat across
/// selectivity, the two tests above could both pass while the search ignored the
/// filter for pruning entirely and always scanned.
#[test]
fn work_scales_with_selectivity() {
    let features = corpus();
    let hnsw = build(&features);
    let qs = queries(30);
    let mut searcher = Searcher::default();

    let measure = |every: usize, searcher: &mut Searcher<u64>| -> f64 {
        let filter = move |id: usize| id.is_multiple_of(every);
        EVALS.with(|e| e.set(0));
        for q in &qs {
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            hnsw.nearest_filtered(q, EF, searcher, &mut dest, &filter);
        }
        EVALS.with(|e| e.get()) as f64 / qs.len() as f64
    };

    // Accept-all versus one-in-fifty. Measured 711.6 versus 3987.5, a ratio of
    // 0.18; the 0.5 bound sits well clear of both that and the 1.0 a
    // filter-ignoring search would produce.
    let permissive = measure(1, &mut searcher);
    let selective = measure(50, &mut searcher);

    assert!(
        permissive / selective < 0.5,
        "an accept-all filter cost {permissive:.1} evals/query and a one-in-fifty \
         filter cost {selective:.1} (ratio {:.2}). Work that does not respond to \
         selectivity means the predicate is not pruning the traversal at all.",
        permissive / selective
    );
}
