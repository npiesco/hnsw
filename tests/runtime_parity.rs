//! Behavioral-parity guard: the runtime-degree `HnswRuntime` MUST produce
//! byte-identical construction + search results to the const-generic
//! `Hnsw<_, _, _, M, M0>` when configured with the SAME `(M, M0)`, the SAME
//! seeded PRNG, and the SAME insertion order.
//!
//! ## Why this is a real behavioral test (not a tripwire)
//!
//! Both indexes are driven through their REAL `insert`/`nearest` paths over an
//! identical seeded corpus. The HNSW graph is fully deterministic given a fixed
//! PRNG sequence + insertion order + neighbor-selection tie-breaks, so a
//! faithful runtime mirror of the const-generic algorithm yields byte-identical
//! `Neighbor { index, distance }` sequences for every query. Any divergence in
//! level assignment, neighbor pruning, layer descent, or tie-breaking makes the
//! search results differ and this test fails RED for that real runtime reason.
//!
//! This is the guard that lets immutlex swap the const-generic chunk index's
//! degrees for an operator-configured runtime `M` on the transient doc-mean
//! index WITHOUT silently changing ANN behavior.

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

/// Deterministic seeded corpus shared by both indexes.
fn corpus() -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(0xA11CE5EED);
    (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect()
}

fn build_const(features: &[Vec<f32>]) -> Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> {
    let mut searcher = Searcher::default();
    // Fixed seed so level assignment is reproducible and matches the runtime index.
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

#[test]
fn runtime_degree_hnsw_matches_const_generic_byte_for_byte() {
    let features = corpus();
    let cst = build_const(&features);
    let rt = build_runtime(&features);

    assert_eq!(cst.len(), rt.len(), "index sizes must match");
    assert_eq!(cst.len(), N);

    // Query with every corpus point plus some fresh random probes; assert the
    // full ranked neighbor list is byte-identical between the two indexes.
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

    let ef = 64;
    let k = 20;
    for (qi, q) in probes.iter().enumerate() {
        let mut cst_searcher = Searcher::default();
        let mut rt_searcher = Searcher::default();
        let mut cst_dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            k
        ];
        let mut rt_dest = vec![
            Neighbor {
                index: !0,
                distance: !0
            };
            k
        ];

        let cst_res = cst
            .nearest(q, ef, &mut cst_searcher, &mut cst_dest)
            .to_vec();
        let rt_res = rt.nearest(q, ef, &mut rt_searcher, &mut rt_dest).to_vec();

        assert_eq!(
            cst_res, rt_res,
            "runtime-degree HNSW diverged from const-generic <{M},{M0}> on probe {qi}: \
             const={cst_res:?} runtime={rt_res:?}",
        );
    }
}
