//! `layer_item_id` / `layer_feature` must actually map a [`Hnsw::search_layer`]
//! result back to its zero-layer node.
//!
//! ## The bug this pins
//!
//! `search_layer(q, ef, level, ..)` returns indices that are local to
//! `self.layers[level - 1]` — its own early return fires at `ix + 1 == level`
//! while iterating `self.layers`, and `layer_len` agrees, reporting
//! `self.layers[level - 1].len()`.
//!
//! `layer_item_id` indexed `self.layers[level]` instead. That is off by one
//! against every other user of `level` in the crate, and it is not a subtle
//! wrong-answer bug — it panics for **every** non-zero level:
//!
//!   * for a mid level, the item index is valid for the intended layer but
//!     usually out of range for the (smaller) layer above it, and
//!   * for the top level, `self.layers[level]` is out of bounds outright, since
//!     the highest valid level equals `self.layers.len()`.
//!
//! Measured on a 2000-point index (3 levels; layer lens 2000 / 156 / 9):
//! `layer_item_id(1, 67)` panicked with "len is 9 but the index is 67", and
//! `layer_item_id(2, 6)` panicked with "len is 2 but the index is 2".
//!
//! This is inherited from upstream `rust-cv/hnsw` — it is not a soft-delete
//! regression.
//!
//! ## Why the assertions are what they are
//!
//! Asserting merely "does not panic" would be satisfied by any in-bounds but
//! semantically wrong mapping. So the sharp assertion is that the feature of the
//! mapped id sits at *exactly* the distance `search_layer` reported: the search
//! computed that distance from the zero-node's own feature, so an off-by-one
//! mapping lands on a different tower and the distances disagree.

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
/// Large enough to build a genuinely multi-level hierarchy (3 levels at M=12).
const N: usize = 2000;
const M: usize = 12;
const M0: usize = 24;

fn build() -> (Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0>, Vec<Vec<f32>>) {
    let mut rng = Pcg64::seed_from_u64(0xDE1E7E);
    let features: Vec<Vec<f32>> = (0..N)
        .map(|_| (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0f32)).collect())
        .collect();
    let mut searcher = Searcher::default();
    let mut hnsw =
        Hnsw::new_params_and_prng(Euclidean, Default::default(), Pcg64::seed_from_u64(42));
    for f in &features {
        hnsw.insert(f.clone(), &mut searcher);
    }
    (hnsw, features)
}

/// Every id `layer_item_id` can produce must be a real zero-layer node, for
/// every level and every item that `layer_len` declares to exist.
///
/// This is the direct contract between `layer_len` (which bounds `item`) and
/// `layer_item_id` (which consumes it). It panicked for all `level > 0`.
#[test]
fn layer_item_id_is_in_range_for_every_declared_layer_item() {
    let (hnsw, _features) = build();

    assert!(
        hnsw.layers() >= 3,
        "fixture must build a multi-level hierarchy to exercise mid AND top \
         levels, got {} level(s)",
        hnsw.layers()
    );

    for level in 0..hnsw.layers() {
        let len = hnsw.layer_len(level);
        assert!(len > 0, "level {} is declared to have items", level);

        for item in 0..len {
            let id = hnsw.layer_item_id(level, item);
            assert!(
                id < hnsw.len(),
                "layer_item_id({level}, {item}) = {id} is not a valid zero-layer \
                 node (index has {} nodes)",
                hnsw.len()
            );
        }
    }
}

/// The mapped node must be the node the search actually scored.
///
/// `search_layer` computes its reported distance from the zero-node's feature,
/// so if `layer_item_id` maps into the wrong layer the distances disagree — this
/// catches an in-bounds-but-wrong mapping that a pure panic check would miss.
#[test]
fn layer_item_id_maps_search_results_to_the_node_that_was_scored() {
    let (hnsw, features) = build();
    let metric = Euclidean;

    // Probe with several queries so this does not hinge on one lucky descent.
    for &qi in &[3usize, 41, 500, 1234, 1999] {
        let q = features[qi].clone();
        let mut searcher = Searcher::default();

        for level in 1..hnsw.layers() {
            let mut dest = vec![
                Neighbor {
                    index: !0,
                    distance: !0
                };
                4
            ];
            let got = hnsw
                .search_layer(&q, 8, level, &mut searcher, &mut dest)
                .to_vec();

            assert!(
                !got.is_empty(),
                "query {}: search_layer returned nothing at level {}",
                qi,
                level
            );

            for hit in &got {
                assert!(
                    hit.index < hnsw.layer_len(level),
                    "query {qi}: search_layer returned item {} at level {level}, \
                     which is outside the {} items layer_len reports",
                    hit.index,
                    hnsw.layer_len(level)
                );

                let id = hnsw.layer_item_id(level, hit.index);
                assert_eq!(
                    metric.distance(&q, &features[id]),
                    hit.distance,
                    "query {qi}: layer_item_id({level}, {}) = {id}, but that \
                     node's distance does not match the distance search_layer \
                     reported — the item was mapped into the wrong layer",
                    hit.index
                );

                assert_eq!(
                    hnsw.layer_feature(level, hit.index),
                    &features[id],
                    "query {qi}: layer_feature({level}, {}) disagrees with the \
                     feature of layer_item_id's own result",
                    hit.index
                );
            }
        }
    }
}

/// Level 0 is the identity mapping, and the zero layer holds every node.
#[test]
fn level_zero_is_the_identity_mapping() {
    let (hnsw, features) = build();

    assert_eq!(hnsw.layer_len(0), N, "the zero layer holds every node");
    for item in [0usize, 1, 999, N - 1] {
        assert_eq!(hnsw.layer_item_id(0, item), item);
        assert_eq!(hnsw.layer_feature(0, item), &features[item]);
    }
}
