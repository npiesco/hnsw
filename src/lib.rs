#![no_std]
extern crate alloc;

mod hnsw;

pub use self::hnsw::*;

use ahash::RandomState;
use alloc::{vec, vec::Vec};
use hashbrown::HashSet;
use space::Neighbor;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Params {
    ef_construction: usize,
    /// Multiplier on the level distribution. See [`Params::level_scale`].
    ///
    /// NOT serialized, deliberately. `Params` is a serialized member of `Hnsw`,
    /// and the wire format is positional under bincode, so adding a field here
    /// shifts every subsequent byte and makes previously written indexes fail to
    /// decode with "unexpected end of file". `#[serde(default)]` does NOT rescue
    /// that: defaults only apply to self-describing formats such as JSON, where
    /// a missing field can be detected by name. A JSON round-trip test passed
    /// and gave false confidence; the bincode fixture is what caught it.
    ///
    /// Skipping is also the right model on the merits: the level scale is
    /// build-time configuration that shapes future insertions, not recovered
    /// graph state. `default = "default_level_scale"` is required alongside
    /// `skip`, because the bare `skip` default for `f64` is `0.0`, which would
    /// collapse the whole hierarchy onto layer zero on every load.
    #[cfg_attr(feature = "serde", serde(skip, default = "default_level_scale"))]
    level_scale: f64,
}

#[cfg(feature = "serde")]
fn default_level_scale() -> f64 {
    1.0
}

/// Hard ceiling on the level any single insertion may be assigned.
///
/// `random_level` draws `-ln(uniform) / ln(M)` and multiplies by the level
/// scale, so a large scale produces a large level, and the index allocates one
/// layer per level. An unvalidated scale therefore allocates without bound: a
/// scale of `5000.0` was measured producing 3364 layers after only ten
/// insertions.
///
/// 64 is chosen so the cap can never bind on a realizable index rather than as
/// a round number. A hierarchy of `L` layers is only useful while `M^L <= N`;
/// even at the smallest sensible `M` of 2 that gives `N >= 2^64`, which exceeds
/// the addressable index size. Any index that could legitimately want a 65th
/// layer cannot fit in memory.
pub(crate) const MAX_LEVEL: usize = 64;

impl Params {
    pub fn new() -> Self {
        Default::default()
    }

    /// This is refered to as `efConstruction` in the paper. This is equivalent to the `ef` parameter passed
    /// to `nearest`, but it is the `ef` used when inserting elements. The higher this is, the more likely the
    /// nearest neighbors in each graph level will be correct, leading to a higher recall rate and speed when
    /// calling `nearest`. This parameter greatly affects the speed of insertion into the HNSW.
    ///
    /// This parameter is probably the only one that in important to tweak.
    ///
    /// Defaults to `400` (overkill for most tasks, but only lower after profiling).
    pub fn ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// Scales the level distribution, controlling how tall the hierarchy grows.
    ///
    /// Levels are drawn from `-ln(uniform) / ln(M)`, which ties the expected
    /// number of layers to `M` alone. This multiplies that draw, so the
    /// hierarchy can be flattened without touching connectivity: a scale below
    /// `1.0` yields fewer layers and a shorter descent per query, at some cost
    /// in recall.
    ///
    /// Defaults to `1.0`, which is exactly the original distribution — the knob
    /// is opt-in and changes nothing until set.
    ///
    /// Values are clamped to be finite and positive; a non-positive or non-finite
    /// scale would collapse every node onto the zero layer.
    ///
    /// The idea is taken from `hnswlib-rs`'s `Hnsw::modify_level_scale`
    /// (`jean-pierreBoth/hnswlib-rs`, `src/hnsw.rs`), reimplemented here against
    /// this crate's `Params` rather than copied.
    pub fn level_scale(mut self, level_scale: f64) -> Self {
        self.level_scale = if level_scale.is_finite() && level_scale > 0.0 {
            level_scale
        } else {
            1.0
        };
        self
    }

    /// The configured level scale.
    pub(crate) fn get_level_scale(&self) -> f64 {
        self.level_scale
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            ef_construction: 400,
            level_scale: 1.0,
        }
    }
}

/// Contains all the state used when searching the HNSW
#[derive(Clone, Debug)]
pub struct Searcher<Metric> {
    candidates: Vec<Neighbor<Metric>>,
    nearest: Vec<Neighbor<Metric>>,
    seen: HashSet<usize, RandomState>,
}

impl<Metric> Searcher<Metric> {
    pub fn new() -> Self {
        Default::default()
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.nearest.clear();
        self.seen.clear();
    }
}

impl<Metric> Default for Searcher<Metric> {
    fn default() -> Self {
        Self {
            candidates: vec![],
            nearest: vec![],
            seen: HashSet::with_hasher(RandomState::with_seeds(0, 0, 0, 0)),
        }
    }
}
