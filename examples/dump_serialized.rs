//! Emits the serialized bytes of a fixed, seeded index to stdout as a length
//! and digest.
//!
//! Exists to compare the output of two commits: run it on each and diff the
//! results. The claim it checks is that canonicalizing the empty-slot sentinel
//! left 64-bit output UNCHANGED, so indexes already written stay readable — a
//! claim originally made from reasoning about `serde`'s `usize` impl rather than
//! from measurement.
//!
//! Measured across the change, on 64-bit:
//!
//! ```text
//! fixint_len=49760  fixint_digest=2097de0ca22b2883
//! varint_len=11948  varint_digest=6e99623aac50911c
//! ```
//!
//! identical before and after. Both encodings are emitted because the crate's
//! own tests use fixint while consumers commonly use the default varint, and a
//! change could in principle affect one and not the other.
//!
//! Run with: `cargo run --release --features serde1 --example dump_serialized`

use bincode::Options;
use hnsw::{Hnsw, Params, Searcher};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use space::Metric;

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

/// FNV-1a, inline so this example needs no extra dependency.
fn digest(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn main() {
    let mut rng = Pcg64::seed_from_u64(0xC0FFEE);
    let mut searcher = Searcher::default();
    let mut hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 12, 24> =
        Hnsw::new_params_and_prng(Euclidean, Params::default(), Pcg64::seed_from_u64(42));

    for _ in 0..200 {
        let v: Vec<f32> = (0..8).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        hnsw.insert(v, &mut searcher);
    }

    let bytes = bincode::DefaultOptions::new()
        .with_little_endian()
        .with_fixint_encoding()
        .serialize(&hnsw)
        .expect("serialize");

    // Also emit under the DEFAULT (varint) options, which is what the consumer
    // actually uses — the two encodings must both be checked.
    let varint = bincode::DefaultOptions::new()
        .serialize(&hnsw)
        .expect("serialize varint");

    println!("fixint_len={}", bytes.len());
    println!("fixint_digest={:016x}", digest(&bytes));
    println!("varint_len={}", varint.len());
    println!("varint_digest={:016x}", digest(&varint));
}
