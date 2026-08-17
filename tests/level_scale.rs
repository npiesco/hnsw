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
//! `Params` is a serialized field of `Hnsw`, so adding a field to it would
//! change every graph blob ever written. The consumer of this crate persists
//! indexes with **bincode**, which is positional: fields carry no names, so a
//! new field shifts every subsequent byte and the decoder runs off the end with
//! "unexpected end of file". `#[serde(default)]` does NOT rescue that — defaults
//! only apply to self-describing formats such as JSON, where a missing field can
//! be recognised by name.
//!
//! An earlier revision of this file tested exactly that with a JSON payload,
//! passed, and was wrong: it proved the property in a format nobody persists
//! with. The field is therefore `#[serde(skip)]`, and the tests below assert the
//! byte layout directly against bincode rather than trusting a named format.

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
/// serialized member of `Hnsw`, so a new field would make every previously
/// written index fail to load.
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

/// THE load-bearing compatibility assertion.
///
/// bincode is positional, so the only thing that keeps previously written
/// indexes loadable is that `Params` still occupies exactly the bytes it used
/// to. Asserting equality against a stand-in struct holding only the original
/// field pins that directly, and fails the moment anyone adds a serialized field
/// here — including by removing the `skip`.
#[cfg(feature = "serde")]
#[test]
fn params_occupy_exactly_the_legacy_bincode_layout() {
    #[derive(serde::Serialize)]
    struct LegacyParams {
        ef_construction: usize,
    }

    let current = bincode::serialize(&Params::new().level_scale(0.35))
        .expect("current params must serialize");
    let legacy = bincode::serialize(&LegacyParams {
        ef_construction: 400,
    })
    .expect("legacy must serialize");

    assert_eq!(
        current, legacy,
        "Params no longer serializes to the pre-level_scale byte layout, so every \
         index written by an earlier build will fail to decode. Note the scale was \
         set to a NON-default 0.35 here: the bytes must be identical regardless of \
         its value, because it must not be on the wire at all."
    );
}

/// A payload written before the field existed must decode, in the positional
/// format that actually matters.
#[cfg(feature = "serde")]
#[test]
fn a_legacy_bincode_params_payload_still_decodes() {
    #[derive(serde::Serialize)]
    struct LegacyParams {
        ef_construction: usize,
    }

    let bytes = bincode::serialize(&LegacyParams {
        ef_construction: 400,
    })
    .unwrap();
    let decoded: Params =
        bincode::deserialize(&bytes).expect("a pre-level_scale payload must still decode");

    // It must arrive at the neutral default rather than `f64`'s `0.0`, which
    // would flatten every future insertion onto layer zero.
    let rebuilt = build(decoded);
    let baseline = build(Params::new());
    assert_eq!(
        rebuilt.layers(),
        baseline.layers(),
        "a legacy payload decoded to a level_scale that changes the hierarchy — \
         `skip` without an explicit `default` yields 0.0, which collapses it"
    );
}

/// An unvalidated scale allocates one layer per level, so a large scale
/// allocates without bound. Measured before the cap: `level_scale(5000.0)`
/// produced 3364 layers after ten insertions.
#[test]
fn an_extreme_level_scale_cannot_allocate_unbounded_layers() {
    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> = Hnsw::new_params_and_prng(
        Euclidean,
        Params::new().level_scale(5000.0),
        Pcg64::seed_from_u64(42),
    );
    for i in 0..10 {
        hnsw.insert(vec![i as f32; DIM], &mut searcher);
    }

    // `layers()` counts the zero layer too, hence the +1.
    assert!(
        hnsw.layers() <= 65,
        "an extreme level scale produced {} layers; the cap is not being applied \
         and a larger scale would exhaust memory",
        hnsw.layers()
    );
}

/// The cap must not bind on ordinary configurations, or it would silently
/// truncate a legitimate hierarchy.
#[test]
fn the_level_cap_does_not_bind_on_a_normal_index() {
    let normal = build(Params::new());
    assert!(
        normal.layers() < 65,
        "a default index already reaches {} layers, so the cap is not purely a \
         safety bound and is reshaping normal graphs",
        normal.layers()
    );
}
