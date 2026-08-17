//! `Params::level_scale` controls how tall the hierarchy grows.
//!
//! ## What this is
//!
//! `random_level` draws from `-ln(uniform) / ln(M)`, which fixes the expected
//! number of layers as a function of `M` alone. hnswlib-rs exposes a scale on
//! that expression (`jean-pierreBoth/hnswlib-rs`, `src/hnsw.rs:876-905`,
//! `Hnsw::modify_level_scale`) so the hierarchy can be flattened independently
//! of the connectivity parameter. Reimplemented here against this crate's
//! `Params` rather than copied.
//!
//! A smaller scale means fewer layers, which means a shorter descent per query.
//! It is a real knob for a workload that is deletion-heavy or where the upper
//! layers cost more than they save.
//!
//! ## The serialization constraint this respects
//!
//! `Params` is a serialized field of `Hnsw`, so adding a field to it changes
//! every graph blob ever written. Two consequences, both asserted below:
//!
//! * the field carries `#[serde(default)]`, so indexes written BEFORE it existed
//!   still deserialize — otherwise a dependency bump would brick every persisted
//!   index; and
//! * the default must reproduce the previous behaviour EXACTLY, so the knob is
//!   provably opt-in rather than a silent change to everyone's graphs.

use hnsw::{Hnsw, Params, Searcher};
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
const N: usize = 3000;
const M: usize = 12;
const M0: usize = 24;

fn corpus() -> Vec<Vec<f32>> {
    let mut rng = Pcg64::seed_from_u64(0xDE1E7E);
    (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect()
}

fn build(params: Params) -> Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new_params_and_prng(Euclidean, params, Pcg64::seed_from_u64(42));
    for f in corpus() {
        hnsw.insert(f.clone(), &mut searcher);
    }
    hnsw
}

/// The default must be indistinguishable from the previous behaviour: same
/// layer count, same per-layer sizes. If this fails, adding the knob silently
/// reshaped every existing user's graph.
#[test]
fn the_default_level_scale_reproduces_the_previous_hierarchy() {
    let plain = build(Params::new());
    let explicit = build(Params::new().level_scale(1.0));

    assert_eq!(
        plain.layers(),
        explicit.layers(),
        "an explicit scale of 1.0 produced a different number of layers than the \
         default, so the default is not 1.0"
    );
    for level in 0..plain.layers() {
        assert_eq!(
            plain.layer_len(level),
            explicit.layer_len(level),
            "layer {level} differs in size between the default and an explicit \
             scale of 1.0"
        );
    }
}

/// A smaller scale must actually flatten the hierarchy. Without this the knob
/// could be accepted, stored, serialized and completely ignored.
#[test]
fn a_smaller_level_scale_produces_a_shallower_hierarchy() {
    let tall = build(Params::new());
    let flat = build(Params::new().level_scale(0.35));

    assert!(
        flat.layers() < tall.layers(),
        "level_scale 0.35 produced {} layers, the same or more than the default's \
         {} — the parameter is not reaching `random_level`",
        flat.layers(),
        tall.layers()
    );

    // Every node must still be present; flattening moves nodes between layers,
    // it does not drop them.
    assert_eq!(
        flat.len(),
        N,
        "flattening the hierarchy lost nodes from the zero layer"
    );
}

/// Flattening must not destroy recall. A knob that halves the layer count and
/// quietly halves accuracy is not a tuning parameter, it is a regression.
#[test]
fn a_flatter_hierarchy_keeps_recall_within_budget() {
    const K: usize = 10;
    const EF: usize = 64;
    /// Predeclared: a flatter hierarchy may cost recall, but not more than this.
    const MAX_RECALL_LOSS: f64 = 0.05;

    let features = corpus();
    let tall = build(Params::new());
    let flat = build(Params::new().level_scale(0.35));

    let mut qrng = Pcg64::seed_from_u64(7);
    let mut tall_hits = 0usize;
    let mut flat_hits = 0usize;
    let mut total = 0usize;

    for _ in 0..100 {
        let q: Vec<f32> = (0..DIM).map(|_| qrng.gen_range(-1.0f32..1.0f32)).collect();

        // Brute-force truth.
        let mut all: Vec<(u64, usize)> = features
            .iter()
            .enumerate()
            .map(|(i, f)| (Euclidean.distance(&q, f), i))
            .collect();
        all.sort();
        let truth: std::collections::BTreeSet<usize> =
            all.into_iter().take(K).map(|(_, i)| i).collect();

        for (hnsw, hits) in [(&tall, &mut tall_hits), (&flat, &mut flat_hits)] {
            let mut s = Searcher::default();
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                K
            ];
            let got = hnsw.nearest(&q, EF, &mut s, &mut dest).to_vec();
            *hits += got.iter().filter(|n| truth.contains(&n.index)).count();
        }
        total += truth.len();
    }

    let tall_recall = tall_hits as f64 / total as f64;
    let flat_recall = flat_hits as f64 / total as f64;
    assert!(
        tall_recall - flat_recall <= MAX_RECALL_LOSS,
        "flattening cost {:.4} recall (from {tall_recall:.4} to {flat_recall:.4}), \
         over the {MAX_RECALL_LOSS} budget",
        tall_recall - flat_recall
    );
}

/// An index serialized WITHOUT the field must still deserialize. `Params` is a
/// serialized member of `Hnsw`, so a new field without `#[serde(default)]` would
/// make every previously written index fail to load.
#[cfg(feature = "serde")]
#[test]
fn params_without_a_level_scale_field_still_deserialize() {
    // The exact shape a pre-`level_scale` build wrote: a struct with only
    // `ef_construction`.
    let legacy = r#"{"ef_construction":400}"#;
    let params: Params =
        serde_json::from_str(legacy).expect("params written before level_scale existed must load");

    // And it must come back with the neutral default, not garbage.
    let rebuilt = build(params);
    let baseline = build(Params::new());
    assert_eq!(
        rebuilt.layers(),
        baseline.layers(),
        "a legacy Params deserialized to a level_scale that changes the hierarchy"
    );
}
