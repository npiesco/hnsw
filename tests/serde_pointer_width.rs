//! The serialized form must not depend on the pointer width of the machine that
//! wrote it.
//!
//! ## The defect this pins
//!
//! An unused neighbour slot is `!0usize` in memory, and `usize` is 64 bits on a
//! server and 32 bits on `wasm32`. Serializing that value directly made the file
//! format machine-dependent in both directions:
//!
//! * a 64-bit writer emitted `18446744073709551615`, which a 32-bit reader
//!   REJECTS — `invalid value: integer 18446744073709551615, expected usize`, so
//!   an index built natively could not be opened in a browser at all;
//! * a 32-bit writer emitted `4294967295`, which a 64-bit reader ACCEPTS as an
//!   ordinary neighbour index, because the `take_while(|&n| n != !0)` terminator
//!   in `get_neighbors` does not match it. That direction corrupts the graph
//!   silently instead of failing.
//!
//! This was invisible from a 64-bit test suite: every assertion below passes
//! trivially on 64-bit both before and after the fix, because the in-memory
//! sentinel and the wire sentinel coincide there. What makes the tests real is
//! that they assert the WIRE BYTES rather than a round trip, so they state the
//! cross-target contract explicitly and fail on a 32-bit target if it breaks.
//! The end-to-end proof is in the consumer: `hnswlib-wasm`'s
//! `tests/hnsw_version_compat.rs` loads a natively written fixture under
//! `wasm32-wasip2` and failed with exactly the error quoted above.

#![cfg(feature = "serde")]

use bincode::Options;
use hnsw::{Hnsw, Params, Searcher};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use space::{Metric, Neighbor};

#[derive(Serialize, Deserialize)]
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

const M: usize = 12;
const M0: usize = 24;

fn options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_little_endian()
        .with_fixint_encoding()
}

fn build(n: usize) -> Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> {
    use rand::{Rng, SeedableRng};
    let mut rng = Pcg64::seed_from_u64(0xC0FFEE);
    let mut searcher = Searcher::default();
    let mut hnsw =
        Hnsw::new_params_and_prng(Euclidean, Params::default(), Pcg64::seed_from_u64(42));
    for _ in 0..n {
        let v: Vec<f32> = (0..8).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        hnsw.insert(v, &mut searcher);
    }
    hnsw
}

/// An unused neighbour slot must appear on the wire as the canonical sentinel.
///
/// A small index guarantees unused slots: with `M0 = 24` and only 5 nodes, no
/// zero-layer neighbour list can be full.
///
/// Scope, stated honestly: this is a PRESENCE check over a whole serialized
/// index, and eight `0xFF` bytes could in principle occur in unrelated state, so
/// it is weak on its own. The unambiguous assertion is the exact-layout unit
/// test beside the implementation in `src/hnsw/serde_impl.rs`, which constructs
/// a `NeighborNodes` directly. What this adds is that the sentinel survives
/// composition into a real index rather than only in isolation.
#[test]
fn an_empty_neighbour_slot_serializes_as_the_canonical_sentinel() {
    let hnsw = build(5);
    let bytes = options().serialize(&hnsw).expect("serialize");

    let sentinel = u64::MAX.to_le_bytes();
    assert!(
        bytes.windows(sentinel.len()).any(|w| w == sentinel),
        "no canonical empty-slot sentinel found in {} serialized bytes",
        bytes.len()
    );
}

/// An index whose empty slots carry the LEGACY 32-bit value must still load and
/// answer identically.
///
/// This is the assertion that actually discriminates. The two tests it replaced
/// — a same-target round trip and a serialize/reload/re-serialize comparison —
/// pass on every target both before and after the fix, because a locally created
/// payload never contains a foreign sentinel. They were false evidence.
///
/// The rewrite direction is fixed rather than target-dependent, because after
/// the fix EVERY target writes the canonical `u64::MAX`; a first version of this
/// test rewrote "the other width's" value and so found nothing to rewrite when
/// run on 32-bit. Substituting `u32::MAX` reproduces exactly what an old 32-bit
/// build emitted, which is the direction that used to corrupt silently: that
/// value is a representable 64-bit `usize`, so a reader without the legacy
/// mapping imports it as a real neighbour index rather than a terminator.
///
/// The opposite direction — a 64-bit writer read by a 32-bit reader — needs no
/// rewriting: it IS the canonical encoding, and running this whole file under
/// `wasm32-wasip2` exercises it.
#[test]
fn an_index_with_legacy_32_bit_sentinels_still_loads_and_answers() {
    let hnsw = build(120);

    let mut searcher = Searcher::default();
    let query: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        10
    ];
    let expected = hnsw.nearest(&query, 64, &mut searcher, &mut dest).to_vec();

    let bytes = options().serialize(&hnsw).expect("serialize");

    let canonical = u64::MAX.to_le_bytes();
    let legacy = (u32::MAX as u64).to_le_bytes();

    let mut rewritten = bytes.clone();
    let mut swaps = 0usize;
    let mut i = 0;
    while i + 8 <= rewritten.len() {
        if rewritten[i..i + 8] == canonical {
            rewritten[i..i + 8].copy_from_slice(&legacy);
            swaps += 1;
            i += 8;
        } else {
            i += 1;
        }
    }
    assert!(
        swaps > 0,
        "the fixture contained no sentinels to rewrite, so this test would prove \
         nothing"
    );

    let reloaded: Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> = options()
        .deserialize(&rewritten)
        .expect("an index with legacy 32-bit sentinels must still load");

    let mut searcher2 = Searcher::default();
    let mut dest2 = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        10
    ];
    let got = reloaded
        .nearest(&query, 64, &mut searcher2, &mut dest2)
        .to_vec();

    assert_eq!(
        expected, got,
        "an index whose empty slots were written by an old 32-bit build returned \
         different neighbours, so its sentinels were imported as real indices \
         instead of being recognised as empty"
    );
}

/// A same-target round trip must of course still work.
///
/// This does NOT evidence the pointer-width fix — it passes identically before
/// and after — but it guards the ordinary path against a regression introduced
/// while changing the encoding.
#[test]
fn a_same_target_round_trip_preserves_search_results() {
    let hnsw = build(200);

    let mut searcher = Searcher::default();
    let query: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
    let mut dest = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        10
    ];
    let before = hnsw.nearest(&query, 64, &mut searcher, &mut dest).to_vec();

    let bytes = options().serialize(&hnsw).expect("serialize");
    let reloaded: Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> =
        options().deserialize(&bytes).expect("deserialize");

    let mut searcher2 = Searcher::default();
    let mut dest2 = vec![
        Neighbor {
            index: !0,
            distance: !0
        };
        10
    ];
    let after = reloaded
        .nearest(&query, 64, &mut searcher2, &mut dest2)
        .to_vec();

    assert_eq!(
        before, after,
        "a serialize/deserialize round trip changed search results, so the \
         reloaded graph is not the graph that was written"
    );
}

/// Re-serializing a reloaded index must reproduce the original bytes exactly.
///
/// Like the round trip above this is not pointer-width evidence — a locally
/// created payload cannot contain a foreign sentinel — but it does pin that
/// decoding and re-encoding are exact inverses, which is what makes the
/// cross-width test above meaningful.
#[test]
fn re_serializing_a_reloaded_index_reproduces_the_bytes() {
    let hnsw = build(50);
    let first = options().serialize(&hnsw).expect("serialize");

    let reloaded: Hnsw<Euclidean, Vec<f32>, Pcg64, M, M0> =
        options().deserialize(&first).expect("deserialize");
    let second = options().serialize(&reloaded).expect("re-serialize");

    assert_eq!(
        first, second,
        "re-serializing a reloaded index produced different bytes"
    );
}
