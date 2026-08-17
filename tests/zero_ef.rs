//! A zero `ef` must return nothing, on the ordinary search path as well as the
//! filtered one.
//!
//! ## The defect this pins
//!
//! The result-set bound is tested as `nearest.len() == cap`. That equality never
//! holds when `cap` is 0, so `nearest` grew without limit: a caller asking for no
//! results received a full `dest` worth of them, after a COMPLETE traversal of
//! the graph.
//!
//! The filtered path was guarded when it was written. The ordinary path was not,
//! and it is the one nearly every caller uses.

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
const N: usize = 500;

fn build() -> (Hnsw<Euclidean, Vec<f32>, Pcg64, 12, 24>, Vec<Vec<f32>>) {
    let mut rng = Pcg64::seed_from_u64(0xEF0);
    let features: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new(Euclidean);
    for f in &features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    (hnsw, features)
}

fn empty_dest(k: usize) -> Vec<Neighbor<u64>> {
    vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        k
    ]
}

#[test]
fn ordinary_nearest_with_a_zero_ef_returns_nothing() {
    let (hnsw, features) = build();
    let mut searcher = Searcher::default();
    let mut dest = empty_dest(10);

    let got = hnsw.nearest(&features[0], 0, &mut searcher, &mut dest);

    assert!(
        got.is_empty(),
        "an ef of 0 returned {} results; the bound is `len() == ef`, which never \
         holds at 0, so the result heap is unbounded",
        got.len()
    );
}

#[test]
fn ordinary_search_layer_with_a_zero_ef_returns_nothing() {
    let (hnsw, features) = build();
    let mut searcher = Searcher::default();
    let mut dest = empty_dest(10);

    let got = hnsw.search_layer(&features[0], 0, 0, &mut searcher, &mut dest);

    assert!(
        got.is_empty(),
        "search_layer with ef 0 returned {} results",
        got.len()
    );
}

/// The guard must not have broken the ordinary case, or it would be a trivially
/// "correct" search that returns nothing at all.
#[test]
fn a_nonzero_ef_still_returns_results() {
    let (hnsw, features) = build();
    let mut searcher = Searcher::default();
    let mut dest = empty_dest(10);

    let got = hnsw.nearest(&features[0], 32, &mut searcher, &mut dest);

    assert_eq!(
        got.len(),
        10,
        "a normal query stopped returning results after the zero-ef guard"
    );
    assert_eq!(
        got[0].index, 0,
        "querying with an indexed vector must return it first"
    );
}
